import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict.tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase, TensorDictSequential, TensorDictModule
from einops.layers.torch import Rearrange
from torchrl.modules import ProbabilisticActor, GRUModule, IndependentNormal
from torchrl.envs.transforms import CatTensors
from trainning_utils import ValueNorm, make_batch, make_mlp, GAE, IndependentBeta, BetaActor, vec_to_world
from profiler import get_profiler

NORM_EPS = 1e-3

class _LidarCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.LazyConv2d(out_channels=4, kernel_size=[5, 3], padding=[2, 1]), nn.ELU(),
            nn.LazyConv2d(out_channels=16, kernel_size=[5, 3], stride=[2, 1], padding=[2, 1]), nn.ELU(),
            nn.LazyConv2d(out_channels=16, kernel_size=[5, 3], stride=[2, 2], padding=[2, 1]), nn.ELU(),
            Rearrange("n c w h -> n (c w h)"),
            nn.LazyLinear(128), nn.LayerNorm(128, eps=NORM_EPS),
        )

    def forward(self, x):
        # x: [B, T, C, W, H] or [N, C, W, H]
        # when forward, combine the batch and time dimensions if input in B&T format
        if x.dim() == 5:
            b, t, c, w, h = x.shape
            x = x.reshape(b * t, c, w, h)
            x = self.net(x)             # [B*T, 128]
            x = x.view(b, t, -1)        # [B, T, 128]
            return x
        elif x.dim() == 4 or x.dim() == 3:
            return self.net(x)          # [N, 128] or [1, 128]
        else:
            raise RuntimeError(f"Unexpected lidar tensor shape: {tuple(x.shape)}")

class ResidualActionModule(nn.Module):
    """
    残差模块，将网络输出的 'delta' 与 'human_action' 相加。
    为了保证加法在同一尺度，需要知道 action_limit。
    """
    def __init__(self, action_limit):
        super().__init__()
        self.action_limit = action_limit

    def forward(self, loc, human_action):
        # 1. Normalize human action to [-1, 1] (假设 human_action 是物理单位的速度)
        # 注意：这里假设 human_action 和 loc 都在 Body Frame
        human_action_norm = human_action / self.action_limit
        
        # 2. Residual Connection: Mean = Network_Output + Human_Action
        # 这里的 loc 是网络学到的“修正量”
        new_loc = loc + human_action_norm
        
        return new_loc

class SplitLayer(nn.Module):
    """
    Split 模块，把TanhNormal网络的输出拆分为 loc 和 scale
    """
    def __init__(self, action_dim):
        super().__init__()
        self.action_dim = action_dim
    def forward(self, x):
        loc, scale = x.split(self.action_dim, dim=-1)
        # 对 scale 进行处理保证其为正数 (softplus)
        scale = torch.nn.functional.softplus(scale) + 1e-4 
        return loc, scale

class SimpleResidualPPO(TensorDictModuleBase):
    def __init__(self, cfg, observation_spec, action_spec, device):
        super().__init__()
        self.cfg = cfg
        self.device = device

        self.using_rnn = cfg.rnn.enable

        # Get obs spec dims
        state_dim = observation_spec["agents", "observation", "state"].shape[-1]
        human_action_dim = observation_spec["agents", "observation", "human_action"].shape[-1]

        # Check if lidar is present in observation spec
        self.has_lidar = "lidar" in observation_spec["agents", "observation"]
        
        modules = []
        cat_keys = []
        
        cnn_feature_dim = 0
        if self.has_lidar:
            # Extract LiDAR Feature
            feature_extractor_network = _LidarCNN().to(self.device)
            modules.append(TensorDictModule(feature_extractor_network, [("agents", "observation", "lidar")], ["_cnn_feature"]))
            cat_keys.append("_cnn_feature")
            cnn_feature_dim = 128

        # Add keys, depending on whether prev_action is included
        cat_keys.extend([
            ("agents", "observation", "state"), 
            ("agents", "observation", "human_action"),
        ])

        if self.using_rnn:
            # ====== add GRU Module in feature_extractor ======
            # TODO: GRU module only embeds the prev states & actions, defined before the current state & human action.
            # RNN network dims for temporal information of observations
            gru_input_dim = cnn_feature_dim + state_dim + human_action_dim
            gru_hidden_dim = cfg.algo.rnn.gru_hidden_dim

            self.gru_num_layers = cfg.algo.rnn.gru_num_layers
            self.gru_hidden_dim = gru_hidden_dim
            self.gru_model = GRUModule(
                    input_size=gru_input_dim,
                    hidden_size=gru_hidden_dim,
                    device=self.device,
                    in_key="_embed",
                    out_key="_embed",
                )
            print("in keys: ", self.gru_model.in_keys)
            print("out keys: ", self.gru_model.out_keys)
            
            # 3. Concat different obs features
            modules.append(CatTensors(
                in_keys=cat_keys, 
                out_key="_embed_inputs",  # Output to intermediate key
                del_keys=False
            ))
            
            # Add LayerNorm to normalize inputs before GRU
            modules.append(TensorDictModule(
                nn.LayerNorm(gru_input_dim, eps=NORM_EPS),
                in_keys=["_embed_inputs"],
                out_keys=["_embed"]
            ))
            
            # 4. Add a GRU network
            modules.append(self.gru_model)
        else:
            # ====== Only concatenate features, no GRU ======
            modules.append(CatTensors(
                in_keys=cat_keys, 
                out_key="_embed",  # Output to intermediate key
                del_keys=False
            ))
        
        modules.append(TensorDictModule(make_mlp([256, 256]), ["_embed"], ["_feature"]))
        
        # Rearrange the Feature Extractor network
        self.feature_extractor = TensorDictSequential(*modules).to(self.device)

        # Actor network, now get input from the GRU output feature
        actual_action_spec = action_spec[("agents", "action")]
        self.n_agents, self.action_dim = actual_action_spec.shape[-2:]
        self.action_limit = cfg.actor.action_limit  # action speed limit

        # actor net 是一个简单的MLP网络，用于预测动作分布的 Mean (loc) 和 Std (scale)
        self.actor_net = TensorDictModule(
            nn.Sequential(
                make_mlp([256, 256]), # TODO: pramalize mlp layers
                nn.Linear(256, self.action_dim * 2) # output [loc, scale]
            ),
            in_keys=["_feature"],
            out_keys=["_actor_logits"]
        ).to(self.device)

        split_module = TensorDictModule(
            SplitLayer(self.action_dim),
            in_keys=["_actor_logits"], # 来自 feature_extractor 的输出
            out_keys=["_loc_delta", "scale"]
        )

        # 残差加法模块，加入观察中的用户指令
        # 输入: _loc_delta (网络生成的修正量), human_action (来自观测)
        # 输出: _loc (最终分布的均值)
        residual_module = TensorDictModule(
            ResidualActionModule(self.action_limit),
            in_keys=["_loc_delta", ("agents", "observation", "human_action")], 
            out_keys=["loc"]
        )

        # 最终的actor网络，
        # TODO: 不使用 TanhNormal 分布，因为无法精确计算均值，而使用蒙特卡洛方法会拖慢训练效率
        self.actor = ProbabilisticActor(
            module=TensorDictSequential(self.actor_net, split_module, residual_module),
            in_keys=["loc", "scale"],
            out_keys=[("agents", "action_normalized")],
            distribution_class=IndependentNormal,
            return_log_prob=True,
            log_prob_key="sample_log_prob"
        ).to(self.device)

        # Critic network
        self.critic = TensorDictModule(
            nn.LazyLinear(1), ["_feature"], ["state_value"] 
        ).to(self.device)
        self.value_norm = ValueNorm(1).to(self.device)

        # Loss related
        self.gae = GAE(0.99, 0.95) # generalized adavantage esitmation
        self.critic_loss_fn = nn.HuberLoss(delta=10, reduction='none') # huberloss (L1+L2): https://pytorch.org/docs/stable/generated/torch.nn.HuberLoss.html

        # Optimizer
        self.feature_extractor_optim = torch.optim.Adam(self.feature_extractor.parameters(), lr=cfg.feature_extractor.learning_rate)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor.learning_rate)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=cfg.actor.learning_rate)

        # Dummy Input for nn lazymodule
        dummy_input = observation_spec.zero()

        if self.using_rnn:
            # Initial values of recurrent_state and is_init for GRU module
            dummy_input.set("is_init", torch.ones(dummy_input.batch_size, dtype=torch.bool, device=self.device))
            dummy_input.set(
                "recurrent_state",
                torch.zeros(
                    (*dummy_input.batch_size, self.gru_num_layers, self.gru_hidden_dim),
                    device=self.device,
                ),
            )
            print("[PPO]dummy_input: ", dummy_input)

        self.__call__(dummy_input)

        # Initialize network
        def init_(module):
            if isinstance(module, nn.Linear):
                # 为配合 Tanh/ReLU，中间层通常使用 sqrt(2) 作为增益 (而不再使用原来的0.01)
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.)

        self.feature_extractor.apply(init_)  # feature_extractor也可以初始化
        self.actor.apply(init_)
        self.critic.apply(init_)

        def init_residual(module):
            if isinstance(module, nn.Linear):
                # 将最后一层的权重初始化得非常小，偏置设为 0
                # 这样初始状态下 _loc_delta ≈ 0 从而 _loc ≈ human_action (恒等映射)
                nn.init.constant_(module.weight, 1e-5)  # 1e-5 / 0.
                nn.init.constant_(module.bias, 0.)
                
        self.actor_net.module[-1].apply(init_residual)  # only init the last linear layer of actor net

    def __call__(self, tensordict):
        self.feature_extractor(tensordict)
        
        # === Probe: Check Feature Extractor Output ===
        if torch.isnan(tensordict.get("_feature")).any():
            print("[PPO Probe] NaN detected in feature extractor output (_feature)!")
            
            if torch.isnan(tensordict.get("_embed")).any():
                 print("[PPO Probe] NaN is coming from GRU output (_embed)!")
            
            if self.using_rnn and torch.isnan(tensordict.get("_embed_inputs")).any():
                 print("[PPO Probe] NaN is coming from Inputs before LayerNorm (_embed_inputs)!")
                 # Check individual components to find the culprit
                 if self.has_lidar and torch.isnan(tensordict.get("_cnn_feature")).any(): print("  -> _cnn_feature is NaN")
                 if torch.isnan(tensordict.get(("agents", "observation", "state"))).any(): print("  -> state is NaN")
            
            raise ValueError("NaN in PPO forward pass")
        # =============================================

        self.actor(tensordict)
        self.critic(tensordict)

        # [OLD]"action_normalized": input action in target frame, range [0, 1]. need to scale to [-action_limit, action_limit]
        # actions = (2 * tensordict["agents", "action_normalized"] * self.cfg.actor.action_limit) - self.cfg.actor.action_limit

        action_norm = tensordict["agents", "action_normalized"]
        action_norm_clamped = action_norm.clamp(-1.0, 1.0)

        # "action_normalized" from TanhNormal is in range [-1, 1]
        actions = action_norm_clamped * self.cfg.actor.action_limit

        # transform to world frame
        actions_world = vec_to_world(
            actions, tensordict["agents", "observation", "state"]
        )

        tensordict["agents", "action"] = actions_world
        # Save a copy as "command" because VelController will overwrite "action" with thrusts
        tensordict["agents", "command"] = actions_world.clone() 
        return tensordict

    def get_recurrent_primer(self):
        """
        Returns a TensorDictPrimer transform that ensures recurrent_state and is_init
        are properly initialized in the environment's TensorDicts.
        """
        if self.using_rnn:
            primer = self.gru_model.make_tensordict_primer()
            return primer
        return None

    def train_op(self, tensordict):
        profiler = get_profiler()
        
        # tensordict: (num_env, num_frames, dim), batchsize = num_env * num_frames
        with profiler.timer("ppo/compute_gae"):
            next_tensordict = tensordict["next"]
            with torch.no_grad():
                # next_tensordict = torch.vmap(self.feature_extractor)(next_tensordict) # calculate features for next state value calculation
                self.feature_extractor(next_tensordict)  # No need to vmap, as the GRU module already handle the (B, T, F) sequence input
                next_values = self.critic(next_tensordict)["state_value"]
            rewards = tensordict["next", "agents", "reward"] # Reward obtained by state transition
            dones = tensordict["next", "terminated"] # Whether the next states are terminal states

            values = tensordict["state_value"] # This is calculated stored when we called forward to obtain actions
            values = self.value_norm.denormalize(values) # denomalize values based on running mean and var of return
            next_values = self.value_norm.denormalize(next_values)

            # OPTIONAL: deal with truncated episodes
            # truncated = tensordict["next", "truncated"]s
            # next_values = torch.where(truncated, values, next_values)

            # calculate GAE: Generalized Advantage Estimation
            adv, ret = self.gae(rewards, dones, values, next_values)

            adv_mean = adv.mean()
            adv_std = adv.std()
            adv = (adv - adv_mean) / adv_std.clip(1e-7)

            self.value_norm.update(ret) # update running mean and var for return
            ret = self.value_norm.normalize(ret)  # normalize return

            tensordict.set("adv", adv)
            tensordict.set("ret", ret)

        # Training
        infos = []
        with profiler.timer("ppo/training_epochs"):
            if self.using_rnn:
                # BPTT training for using RNN network
                for epoch in range(self.cfg.training_epoch_num):

                    batch, t = tensordict.batch_size  # batch = num_envs, t = training_frame_num
                    # only shuffle the env batch, but do not shuffle the time dimension
                    perm = torch.randperm(batch, device=self.device)
                    shuffled_tensordict = tensordict[perm]

                    t_chunk = t // self.cfg.num_minibatches
                    if t_chunk == 0:
                        raise ValueError(f"num_minibatches is larger than the number of frames collected per env. batch:{batch}, training_frame_num:{t}, num_minibatches:{self.cfg.num_minibatches}")
                    for i in range(0, t, t_chunk):
                        if i + t_chunk > t:
                            continue  # drop the last incomplete chunk (TODO: check if need padding)
                        minibatch = shuffled_tensordict[:, i : i+t_chunk]
                        infos.append(self._update(minibatch))
            else:
                # Standard PPO training without RNN
                for epoch in range(self.cfg.training_epoch_num):
                    batch = make_batch(tensordict, self.cfg.num_minibatches)
                    for minibatch in batch:
                        infos.append(self._update(minibatch))

        infos = torch.stack(infos).to_tensordict()
        
        infos = infos.apply(torch.mean, batch_size=[])
        return {k: v.item() for k, v in infos.items()}    

    
    def _update(self, minibatch): # tensordict now is minibatch shape (minibatch_size, t_chunk, ...)
        profiler = get_profiler()
        
        with profiler.timer("ppo/update/forward"):
            self.feature_extractor(minibatch)

            # Get action from the current policy
            action_dist = self.actor.get_dist(minibatch) # this does an actor forward to get "loc" and "scale" and use them to build multivariate normal distribution
            
            log_probs = action_dist.log_prob(
                minibatch[("agents", "action_normalized")]) # based on the gaussian, we can calculate the log prob of the action from the current policy

        with profiler.timer("ppo/update/loss_compute"):
            # Entropy Loss
            # action_entropy = action_dist.entropy()
            # TanhNormal distribution specific: entropy is not implemented, use Monte-Carlo estimate
            action_entropy = action_dist.entropy()
            entropy_loss = -self.cfg.entropy_loss_coefficient * torch.mean(action_entropy)

            # Actor Loss
            advantage = minibatch["adv"] # the advantage is calculated based on GAE in hte previous step
            ratio = torch.exp(log_probs - minibatch["sample_log_prob"]).unsqueeze(-1)
            surr1 = advantage * ratio
            surr2 = advantage * ratio.clamp(1.-self.cfg.actor.clip_ratio, 1.+self.cfg.actor.clip_ratio)
            actor_loss = -torch.mean(torch.min(surr1, surr2)) * self.action_dim 

            # Critic Loss 
            b_value = minibatch["state_value"]
            ret = minibatch["ret"] # Return G
            value = self.critic(minibatch)["state_value"] 
            value_clipped = b_value + (value - b_value).clamp(-self.cfg.critic.clip_ratio, self.cfg.critic.clip_ratio) # this guarantee that critic update is clamped
            critic_loss_clipped = self.critic_loss_fn(ret, value_clipped)
            critic_loss_original = self.critic_loss_fn(ret, value)
            critic_loss = torch.mean(torch.max(critic_loss_clipped, critic_loss_original))

            # Total Loss
            loss = entropy_loss + actor_loss + critic_loss

        with profiler.timer("ppo/update/backward"):
            # Optimize
            self.feature_extractor_optim.zero_grad()
            self.actor_optim.zero_grad()
            self.critic_optim.zero_grad()
            loss.backward()

            # gradient clipping
            # === If using RNN: Add gradient clipping for feature extractor ===
            if self.using_rnn:
                feature_extractor_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.feature_extractor.parameters(), max_norm=5.)
            actor_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.actor.parameters(), max_norm=5.) # to prevent gradient growing too large
            critic_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.critic.parameters(), max_norm=5.)
            
            if self.using_rnn:
                self.feature_extractor_optim.step()
                self.actor_optim.step()
                self.critic_optim.step()
            
                explained_var = 1 - F.mse_loss(value, ret) / ret.var()
                return TensorDict({
                    "actor_loss": actor_loss,
                    "critic_loss": critic_loss,
                    "entropy": entropy_loss,
                    "actor_grad_norm": actor_grad_norm,
                    "critic_grad_norm": critic_grad_norm,
                    "feature_extractor_grad_norm": feature_extractor_grad_norm,
                    "explained_var": explained_var
                }, [])
            else:
                self.actor_optim.step()
                self.critic_optim.step()
            
                explained_var = 1 - F.mse_loss(value, ret) / ret.var()
            return TensorDict({
                "actor_loss": actor_loss,
                "critic_loss": critic_loss,
                "entropy": entropy_loss,
                "actor_grad_norm": actor_grad_norm,
                "critic_grad_norm": critic_grad_norm,
                "explained_var": explained_var
            }, [])

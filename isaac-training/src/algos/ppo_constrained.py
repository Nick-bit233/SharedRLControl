import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict.tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase, TensorDictSequential, TensorDictModule
from einops.layers.torch import Rearrange
from torchrl.modules import ProbabilisticActor, GRUModule
try:
    # TorchRL version dependent
    from torchrl.modules import TanhNormal
except ImportError:  # pragma: no cover
    try:
        from torchrl.modules.distributions import TanhNormal
    except ImportError:
        from torchrl.modules.distributions.continuous import TanhNormal
from torchrl.envs.transforms import CatTensors

from src.core.trainning_utils import ValueNorm, make_batch, make_mlp, GAE, vec_to_world
from src.core.profiler import get_profiler

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
        self.register_buffer("residual_scale", torch.tensor(1.0))

    def forward(self, loc, human_action):
        # 1) Normalize human action to (-1, 1) in action space
        # Note: for TanhNormal, `loc` lives in the pre-tanh space.
        human_action_norm = human_action / self.action_limit
        # clamp to avoid atanh(±1) -> inf
        eps = 1e-6
        human_action_norm = human_action_norm.clamp(-1.0 + eps, 1.0 - eps)

        # 2) Map human action into pre-tanh space so that tanh(loc) ≈ human_action_norm
        human_action_pre_tanh = torch.atanh(human_action_norm)

        # 3) Residual connection in pre-tanh space
        new_loc = (loc * self.residual_scale) + human_action_pre_tanh

        return new_loc

    def set_scale(self, scale):
        self.residual_scale.fill_(scale)

class SplitLayer(nn.Module):
    """
    Split 模块，把 actor 网络的输出拆分为 loc 和 scale
    """
    def __init__(self, action_dim):
        super().__init__()
        self.action_dim = action_dim
    def forward(self, x):
        loc, scale = x.split(self.action_dim, dim=-1)
        # 对 scale 进行处理保证其为正数 (softplus)
        scale = torch.nn.functional.softplus(scale) + 1e-4 
        return loc, scale

class ConstrainedResidualPPO(TensorDictModuleBase):
    def __init__(self, cfg, observation_spec, action_spec, device):
        super().__init__()
        self.cfg = cfg
        self.device = device

        self.using_rnn = cfg.rnn.enable
        
        # === Residual Regularization ===
        # Fixed coefficient for residual magnitude penalty.
        # Keeps residuals from exploding but does NOT compete with actor_loss
        # via a convex combination (which caused the 80% collision plateau).
        # Can be increased later via curriculum to enforce human-following.
        self.reg_coeff = cfg.get("reg_coeff", 0.01)
        # ===============================

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
                out_key="_embed_inputs",  # Output to intermediate key
                del_keys=False
            ))
            
            input_dim = cnn_feature_dim + state_dim + human_action_dim
            modules.append(TensorDictModule(
                nn.LayerNorm(input_dim, eps=NORM_EPS),
                in_keys=["_embed_inputs"],
                out_keys=["_embed"]
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
                make_mlp([256, 256]), 
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
        self.residual_action_module = ResidualActionModule(self.action_limit)
        residual_module = TensorDictModule(
            self.residual_action_module,
            in_keys=["_loc_delta", ("agents", "observation", "human_action")], 
            out_keys=["loc"]
        )

        # 最终的actor网络
        self.actor = ProbabilisticActor(
            module=TensorDictSequential(self.actor_net, split_module, residual_module),
            in_keys=["loc", "scale"],
            out_keys=[("agents", "action_normalized")],
            distribution_class=TanhNormal,
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
        self.critic_loss_fn = nn.HuberLoss(delta=10, reduction='none') 

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

        self.__call__(dummy_input)

        # Initialize network
        def init_(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.)

        self.feature_extractor.apply(init_) 
        self.actor.apply(init_)
        self.critic.apply(init_)

        def init_residual(module):
            if isinstance(module, nn.Linear):
                # Init residual to be small
                nn.init.constant_(module.weight, 1e-6)
                nn.init.constant_(module.bias, 0.)
                
        self.actor_net.module[-1].apply(init_residual)  # only init the last linear layer of actor net
    
    def set_reg_coeff(self, value):
        """Set the residual regularization coefficient (for curriculum)."""
        self.reg_coeff = value

    def set_residual_scale(self, scale):
        """Set the scale of the residual policy output (0.0 to 1.0)"""
        if hasattr(self, "residual_action_module"):
            self.residual_action_module.set_scale(scale)

    def __call__(self, tensordict):
        # === Probe: Check Feature Extractor Output ===
        # Check Inputs (to see if Environment/Simulation produced NaNs)
        obs_state = tensordict.get(("agents", "observation", "state"), None)
        if obs_state is not None and torch.isnan(obs_state).any():
             print(f"[PPO Debug] NaN in State Input: {obs_state}")
             raise ValueError("NaN detected in Input: State")

        obs_human = tensordict.get(("agents", "observation", "human_action"), None)
        if obs_human is not None and torch.isnan(obs_human).any():
             print(f"[PPO Debug] NaN in Human Action Input")
             raise ValueError("NaN detected in Input: Human Action")

        if self.has_lidar:
             obs_lidar = tensordict.get(("agents", "observation", "lidar"), None)
             if obs_lidar is not None and torch.isnan(obs_lidar).any():
                 print(f"[PPO Debug] NaN in Lidar Input")
                 raise ValueError("NaN detected in Input: Lidar")

        self.feature_extractor(tensordict)
        
        # Check Outputs
        if torch.isnan(tensordict.get("_feature")).any():
            # If inputs are clean but output is NaN, check feature extractor weights
            for name, param in self.feature_extractor.named_parameters():
                 if torch.isnan(param).any():
                      raise ValueError(f"NaN detected in Feature Extractor Weights: {name}")
            raise ValueError("NaN in Feature Extractor Output (Weights OK, Inputs OK -> Check Layers/Normalization)")
        # =============================================

        self.actor(tensordict)
        self.critic(tensordict)

        # NOTE:
        # - With TanhNormal, sampled actions are already bounded in (-1, 1).
        # - Do NOT clamp after sampling, otherwise the executed action differs from the
        #   action used to compute `sample_log_prob` (breaks PPO ratios).
        action_norm = tensordict["agents", "action_normalized"]
        actions = action_norm * self.cfg.actor.action_limit

        # transform to world frame (lock roll/pitch, only yaw)
        actions_world = vec_to_world(
            actions, tensordict["agents", "observation", "state"], yaw_only=True
        )

        tensordict["agents", "action"] = actions_world
        # Save a copy as "command" because VelController will overwrite "action" with thrusts
        tensordict["agents", "command"] = actions_world.clone() 
        return tensordict

    def get_recurrent_primer(self):
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
                self.feature_extractor(next_tensordict) 
                next_values = self.critic(next_tensordict)["state_value"]
            rewards = tensordict["next", "agents", "reward"] # Reward obtained by state transition
            dones = tensordict["next", "terminated"] # Whether the next states are terminal states

            values = tensordict["state_value"] 
            values = self.value_norm.denormalize(values)
            next_values = self.value_norm.denormalize(next_values)

            # calculate GAE
            adv, ret = self.gae(rewards, dones, values, next_values)

            adv_mean = adv.mean()
            adv_std = adv.std()
            adv = (adv - adv_mean) / adv_std.clip(1e-7)

            self.value_norm.update(ret) 
            ret = self.value_norm.normalize(ret)  

            tensordict.set("adv", adv)
            tensordict.set("ret", ret)

        # Training
        infos = []
        with profiler.timer("ppo/training_epochs"):
            if self.using_rnn:
                for epoch in range(self.cfg.training_epoch_num):
                    batch, t = tensordict.batch_size
                    perm = torch.randperm(batch, device=self.device)
                    shuffled_tensordict = tensordict[perm]

                    t_chunk = t // self.cfg.num_minibatches
                    if t_chunk == 0:
                        raise ValueError(f"num_minibatches is larger than the number of frames collected per env. batch:{batch}, training_frame_num:{t}, num_minibatches:{self.cfg.num_minibatches}")
                    for i in range(0, t, t_chunk):
                        if i + t_chunk > t:
                            continue 
                        minibatch = shuffled_tensordict[:, i : i+t_chunk]
                        infos.append(self._update(minibatch))
            else:
                for epoch in range(self.cfg.training_epoch_num):
                    batch = make_batch(tensordict, self.cfg.num_minibatches)
                    for minibatch in batch:
                        infos.append(self._update(minibatch))

        infos = torch.stack(infos).to_tensordict()
        
        infos = infos.apply(torch.mean, batch_size=[])
        return {k: v.item() for k, v in infos.items()}    

    
    def _update(self, minibatch): 
        profiler = get_profiler()
        
        with profiler.timer("ppo/update/forward"):
            self.feature_extractor(minibatch)

            # Get action from the current policy
            # This will update "_loc_delta" in minibatch to new values
            action_dist = self.actor.get_dist(minibatch) 

            action = minibatch[("agents", "action_normalized")]
            # Clamp actions away from tanh boundaries to prevent atanh(±1) → ±inf in log_prob
            action_safe = action.clamp(-1.0 + 1e-5, 1.0 - 1e-5)
            log_probs = action_dist.log_prob(action_safe)
            # Some distributions return per-dimension log-probs. PPO expects a single
            # log-prob per (agent, timestep) action vector.
            if log_probs.shape == action_safe.shape:
                log_probs = log_probs.sum(dim=-1)
            # Clamp log_probs to prevent -inf from corrupting gradients
            log_probs = log_probs.clamp(min=-20.0)

        with profiler.timer("ppo/update/loss_compute"):
            # 1. Entropy Loss (Monte Carlo)
            # TanhNormal may not implement analytic entropy; use H ≈ -E[log π(a|s)].
            # We sample once from the current policy to keep the estimator unbiased.
            try:
                entropy_action = action_dist.rsample()
            except AttributeError:
                entropy_action = action_dist.sample()
            # Clamp sampled action to avoid atanh boundary issues in log_prob
            entropy_action = entropy_action.clamp(-1.0 + 1e-5, 1.0 - 1e-5)
            entropy_log_prob = action_dist.log_prob(entropy_action)
            if entropy_log_prob.shape == entropy_action.shape:
                entropy_log_prob = entropy_log_prob.sum(dim=-1)
            # Clamp to prevent -inf
            entropy_log_prob = entropy_log_prob.clamp(min=-20.0)
            entropy_est = -entropy_log_prob
            entropy_loss = -self.cfg.entropy_loss_coefficient * torch.mean(entropy_est)

            # 2. Actor Loss
            advantage = minibatch["adv"] 
            old_log_probs = minibatch["sample_log_prob"]
            if old_log_probs.shape == action.shape:
                old_log_probs = old_log_probs.sum(dim=-1)
            # Clamp old_log_probs as well for consistency
            old_log_probs = old_log_probs.clamp(min=-20.0)
            ratio = torch.exp(log_probs - old_log_probs).unsqueeze(-1)
            # Clamp ratio to prevent extreme values from corrupting gradients
            ratio = ratio.clamp(max=10.0)
            surr1 = advantage * ratio
            surr2 = advantage * ratio.clamp(1.-self.cfg.actor.clip_ratio, 1.+self.cfg.actor.clip_ratio)
            actor_loss = -torch.mean(torch.min(surr1, surr2)) * self.action_dim 

            # 3. Regularization Loss (Minimizing Residual Magnitude)
            # Fixed small coefficient — prevents residual from exploding,
            # but does NOT block the actor from producing large corrections for safety.
            reg_loss = minibatch["_loc_delta"].pow(2).sum(dim=-1).mean()

            # 4. Policy Loss = actor_loss + fixed reg penalty (no convex combination)
            loss_pi = actor_loss + self.reg_coeff * reg_loss
            
            # 5. Critic Loss 
            b_value = minibatch["state_value"]
            ret = minibatch["ret"] 
            value = self.critic(minibatch)["state_value"] 
            value_clipped = b_value + (value - b_value).clamp(-self.cfg.critic.clip_ratio, self.cfg.critic.clip_ratio) 
            critic_loss_clipped = self.critic_loss_fn(ret, value_clipped)
            critic_loss_original = self.critic_loss_fn(ret, value)
            critic_loss = torch.mean(torch.max(critic_loss_clipped, critic_loss_original))

            # Total Loss
            loss = entropy_loss + loss_pi + critic_loss

        with profiler.timer("ppo/update/backward"):
            # Optimize Policy & Critic
            self.feature_extractor_optim.zero_grad()
            self.actor_optim.zero_grad()
            self.critic_optim.zero_grad()
            loss.backward()

            # grad clipping
            feature_extractor_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.feature_extractor.parameters(), max_norm=5.)
            actor_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.actor.parameters(), max_norm=5.) 
            critic_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.critic.parameters(), max_norm=5.)
            
            # optim step
            self.feature_extractor_optim.step()
            self.actor_optim.step()
            self.critic_optim.step()

            explained_var = 1 - F.mse_loss(value, ret) / ret.var()
            
            # Return metrics
            return TensorDict({
                "actor_loss": actor_loss,
                "critic_loss": critic_loss,
                "entropy": entropy_loss,
                "reg_loss": reg_loss,
                "actor_grad_norm": actor_grad_norm,
                "critic_grad_norm": critic_grad_norm,
                "explained_var": explained_var
            }, [])

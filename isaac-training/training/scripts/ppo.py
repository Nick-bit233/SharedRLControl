import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict.tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase, TensorDictSequential, TensorDictModule
from einops.layers.torch import Rearrange
from torchrl.modules import ProbabilisticActor, GRUModule
from torchrl.envs.transforms import CatTensors
from utils import ValueNorm, make_mlp, GAE, IndependentBeta, BetaActor, vec_to_world

class _LidarCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.LazyConv2d(out_channels=4, kernel_size=[5, 3], padding=[2, 1]), nn.ELU(),
            nn.LazyConv2d(out_channels=16, kernel_size=[5, 3], stride=[2, 1], padding=[2, 1]), nn.ELU(),
            nn.LazyConv2d(out_channels=16, kernel_size=[5, 3], stride=[2, 2], padding=[2, 1]), nn.ELU(),
            Rearrange("n c w h -> n (c w h)"),
            nn.LazyLinear(128), nn.LayerNorm(128),
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
        
class _DynObsMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 使用 LazyLinear 适配任意 (1 * dyn_obs_num * 10) 输入维度
        self.net = nn.Sequential(
            Rearrange("n c w h -> n (c w h)"),  # 展平到 [N, C*W*H]
            make_mlp([128, 64])
        )

    def forward(self, x):
        # x: [B, T, C=1, W=dyn_obs_num, H=10] or [N, 1, W, H]
        # same as _LidarCNN, combine batch and time dimensions if input in B&T format
        if x.dim() == 5:
            b, t, c, w, h = x.shape
            x = x.reshape(b * t, c, w, h)
            x = self.net(x)             # [B*T, 64]
            x = x.view(b, t, -1)        # [B, T, 64]
            return x
        elif x.dim() == 4:
            return self.net(x)          # [N, 64]
        else:
            raise RuntimeError(f"Unexpected dyn-obs tensor shape: {tuple(x.shape)}")


class PPO(TensorDictModuleBase):
    def __init__(self, cfg, observation_spec, action_spec, device):
        super().__init__()
        self.cfg = cfg
        self.device = device

        # Get obs spec dims
        state_dim = observation_spec["agents", "observation", "state"].shape[-1]
        dyn_obs_feature_dim = 64
        human_action_dim = observation_spec["agents", "observation", "human_action"].shape[-1]
        prev_action_dim = observation_spec["agents", "observation", "prev_action"].shape[-1]
        
        # 1. Extract LiDAR Feature
        feature_extractor_network = _LidarCNN().to(self.device)
        
        # 2. Extract Dynamic obstacle Feature
        dynamic_obstacle_network = _DynObsMLP().to(self.device)
        
        # RNN network dims for temporal information of observations
        cnn_feature_dim = 128
        # 128(cnn_feature) + 64（dyn_obs_mlp) + 10 + 4 + 4 = 210
        gru_input_dim = cnn_feature_dim + dyn_obs_feature_dim + state_dim + human_action_dim + prev_action_dim
        gru_hidden_dim = 256 # TODO: 可以调整的超参数

        self.gru_num_layers = 1
        self.gru_hidden_dim = gru_hidden_dim
        self.gru_model = GRUModule(
                input_size=gru_input_dim,
                hidden_size=gru_hidden_dim,
                num_layers=self.gru_num_layers,
                in_keys=["_feature_cat", "recurrent_state"],
                out_keys=["_gru_out", ("next", "recurrent_state")],
            )
        
        # Rearrange the Feature Extractor network, include a new GRU module.
        self.feature_extractor = TensorDictSequential(
            TensorDictModule(feature_extractor_network, [("agents", "observation", "lidar")], ["_cnn_feature"]),
            TensorDictModule(dynamic_obstacle_network, [("agents", "observation", "dynamic_obstacle")], ["_dynamic_obstacle_feature"]),
            # 3. Concat different obs features
            CatTensors(
                in_keys=[
                    "_cnn_feature", 
                    "_dynamic_obstacle_feature",
                    ("agents", "observation", "state"), 
                    ("agents", "observation", "human_action"),
                    ("agents", "observation", "prev_action")
                ], 
                out_key="_feature_cat",  # Concat a new observation feature contain human actions
                del_keys=False
            ),  
            # 4. Add a GRU network, accept "_feature_cat" as input
            self.gru_model,
            # 5. Final fusion MLP
            TensorDictModule(make_mlp([256, 256]), ["_gru_out"], ["_feature"]),
        ).to(self.device)

        # Actor network, now get input from the GRU output feature
        self.n_agents, self.action_dim = action_spec.shape
        self.actor = ProbabilisticActor(
            TensorDictModule(BetaActor(self.action_dim), ["_feature"], ["alpha", "beta"]),
            in_keys=["alpha", "beta"],  # Use beta distribution for bounded action space
            out_keys=[("agents", "action_normalized")], 
            distribution_class=IndependentBeta,
            return_log_prob=True
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
        # Because of GRUModule, we need to set initial values for recurrent_state and is_init
        dummy_input.set("is_init", torch.ones(dummy_input.batch_size, dtype=torch.bool, device=self.device))
        dummy_input.set(
            "recurrent_state",
            torch.zeros(
                (*dummy_input.batch_size, self.gru_num_layers, self.gru_hidden_dim),
                device=self.device,
            ),
        )
        # print("[PPO]dummy_input: ", dummy_input)

        self.__call__(dummy_input)

        # Initialize network
        def init_(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 0.01)
                nn.init.constant_(module.bias, 0.)
        self.actor.apply(init_)
        self.critic.apply(init_)

    def __call__(self, tensordict):
        # try:
        #     fc = tensordict.get("_feature_cat", None)
        #     rs = tensordict.get("recurrent_state", None)
        #     if fc is not None:
        #         print(f"[DEBUG] _feature_cat shape: {tuple(fc.shape)}")
        #     if rs is not None:
        #         print(f"[DEBUG] recurrent_state shape: {tuple(rs.shape)}")
        # except Exception as e:
        #     print("[DEBUG] shape debug failed:", e)
        self.feature_extractor(tensordict)
        self.actor(tensordict)
        self.critic(tensordict)

        # Cooridnate change: transform local to world (no need transform Cooridnate as no target is provided.)
        # "action_normalized": input action in target frame, range [0, 1]. need to scale to [-action_limit, action_limit]
        actions = (2 * tensordict["agents", "action_normalized"] * self.cfg.actor.action_limit) - self.cfg.actor.action_limit
        # # transform to world frame (no need now)
        # actions_world = vec_to_world(actions, tensordict["agents", "observation", "direction"]) # transform to world frame
        tensordict["agents", "action"] = actions
        return tensordict

    def get_recurrent_primer(self):
        """
        Returns a TensorDictPrimer transform that ensures recurrent_state and is_init
        are properly initialized in the environment's TensorDicts.
        """
        primer = self.gru_model.make_tensordict_primer()
        return primer

    def train(self, tensordict):
        # tensordict: (num_env, num_frames, dim), batchsize = num_env * num_frames
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

        # calculate GAE: Generalized Advantage Estimation
        adv, ret = self.gae(rewards, dones, values, next_values)
        adv_mean = adv.mean()
        adv_std = adv.std()
        adv = (adv - adv_mean) / adv_std.clip(1e-7)
        self.value_norm.update(ret) # update running mean and var for return
        ret = self.value_norm.normalize(ret)  # normalize return
        tensordict.set("adv", adv)
        tensordict.set("ret", ret)

        # Training: Changed, using BPTT
        infos = []
        for epoch in range(self.cfg.training_epoch_num):

            # --- old implementation ---
            # batch = make_batch(tensordict, self.cfg.num_minibatches)
            # for minibatch in batch:
            #     infos.append(self._update(minibatch))

            batch, t = tensordict.batch_size  # batch = num_envs, t = training_frame_num
            # only shuffle the env batch, but do not shuffle the time dimension
            perm = torch.randperm(batch, device=self.device)
            shuffled_tensordict = tensordict[perm]

            t_chunk = t // self.cfg.num_minibatches
            if t_chunk == 0:
                raise ValueError("num_minibatches is larger than the number of frames collected per env.")
            for i in range(0, t, t_chunk):
                if i + t_chunk > t:
                    continue  # drop the last incomplete chunk (TODO: check if need padding)
                minibatch = shuffled_tensordict[:, i : i+t_chunk]
                infos.append(self._update(minibatch))

        infos = torch.stack(infos).to_tensordict()
        
        infos = infos.apply(torch.mean, batch_size=[])
        return {k: v.item() for k, v in infos.items()}    

    
    def _update(self, minibatch): # tensordict now is minibatch shape (minibatch_size, t_chunk, ...)
        self.feature_extractor(minibatch)

        # Get action from the current policy
        action_dist = self.actor.get_dist(minibatch) # this does an actor forward to get "loc" and "scale" and use them to build multivariate normal distribution
        
        log_probs = action_dist.log_prob(
            minibatch[("agents", "action_normalized")]) # based on the gaussian, we can calculate the log prob of the action from the current policy

        # Entropy Loss
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

        # Optimize
        self.feature_extractor_optim.zero_grad()
        self.actor_optim.zero_grad()
        self.critic_optim.zero_grad()
        loss.backward()

        actor_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.actor.parameters(), max_norm=5.) # to prevent gradient growing too large
        critic_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.critic.parameters(), max_norm=5.)
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
            "explained_var": explained_var
        }, [])
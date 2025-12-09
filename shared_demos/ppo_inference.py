import torch
import torch.nn as nn
from tensordict.nn import TensorDictModuleBase, TensorDictSequential, TensorDictModule
from einops.layers.torch import Rearrange
from torchrl.modules import ProbabilisticActor, GRUModule
from torchrl.envs.transforms import CatTensors
from utils_inference import ValueNorm, make_mlp, IndependentBeta, BetaActor

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

class SimplePPO(TensorDictModuleBase):
    def __init__(self, cfg, observation_spec, action_spec, device):
        super().__init__()
        self.cfg = cfg
        self.device = device

        self.using_rnn = cfg.rnn.enable

        # Get obs spec dims
        state_dim = observation_spec["agents", "observation", "state"].shape[-1]
        human_action_dim = observation_spec["agents", "observation", "human_action"].shape[-1]
        prev_action_dim = observation_spec["agents", "observation", "prev_action"].shape[-1]
        
        # Check if lidar is present in observation spec
        self.has_lidar = "lidar" in observation_spec["agents", "observation"]
        
        modules = []
        cat_keys = []
        
        cnn_feature_dim = 0
        if self.has_lidar:
            # 1. Extract LiDAR Feature
            feature_extractor_network = _LidarCNN().to(self.device)
            modules.append(TensorDictModule(feature_extractor_network, [("agents", "observation", "lidar")], ["_cnn_feature"]))
            cat_keys.append("_cnn_feature")
            cnn_feature_dim = 128

        if self.using_rnn:
            # Add prev_action keys to concatenation
            cat_keys.extend([
                ("agents", "observation", "state"), 
                ("agents", "observation", "human_action"),
                ("agents", "observation", "prev_action")
            ])

            # ====== add GRU Module in feature_extractor ======
            # RNN network dims for temporal information of observations
            gru_input_dim = cnn_feature_dim + state_dim + human_action_dim + prev_action_dim
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
            cat_keys.extend([
                ("agents", "observation", "state"), 
                ("agents", "observation", "human_action"),
            ])

            # ====== Only concatenate features, no GRU ======
            modules.append(CatTensors(
                in_keys=cat_keys, 
                out_key="_embed",  # Output to intermediate key
                del_keys=False
            ))
        
        # 5. Final fusion MLP
        modules.append(TensorDictModule(make_mlp([256, 256]), ["_embed"], ["_feature"]))
        
        # Rearrange the Feature Extractor network
        self.feature_extractor = TensorDictSequential(*modules).to(self.device)

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
                nn.init.orthogonal_(module.weight, 0.01)
                nn.init.constant_(module.bias, 0.)
        
        self.actor.apply(init_)
        self.critic.apply(init_)

    def __call__(self, tensordict):
        self.feature_extractor(tensordict)
        self.actor(tensordict)
        self.critic(tensordict)

        # Cooridnate change: transform local to world (no need transform Cooridnate as no target is provided.)
        # "action_normalized": input action in target frame, range [0, 1]. need to scale to [-action_limit, action_limit]
        actions = (2 * tensordict["agents", "action_normalized"] * self.cfg.actor.action_limit) - self.cfg.actor.action_limit
        tensordict["agents", "action"] = actions
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

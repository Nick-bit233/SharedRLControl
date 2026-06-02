"""
Constrained Residual PPO with Beta Distribution.

Drop-in replacement for ppo_constrained.py (TanhNormal version).
Key differences:
  - Actor outputs mean_delta + concentration instead of loc + scale
  - Residual connection in action space [0,1] instead of pre-tanh space
  - Beta distribution: naturally bounded, analytic entropy, controllable variance
  - No tanh Jacobian → cleaner log_prob, no boundary clamp hacks

Usage: swap import in train.py:
    from src.algos.ppo_constrained_beta import ConstrainedResidualPPO_Beta as ConstrainedResidualPPO
"""

from collections import OrderedDict
from itertools import chain

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from tensordict.tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase, TensorDictSequential, TensorDictModule
from einops.layers.torch import Rearrange
from torchrl.modules import ProbabilisticActor, GRUModule
from torchrl.envs.transforms import CatTensors

from src.core.trainning_utils import ValueNorm, make_batch, make_mlp, GAE, vec_to_world
from src.core.profiler import get_profiler

NORM_EPS = 1e-3


# ============================================================
# Beta Distribution Wrapper for ProbabilisticActor
# ============================================================

class ScaledBeta(D.TransformedDistribution):
    """Beta distribution scaled to [-1, 1] for use with ProbabilisticActor.
    
    Internally uses Beta(alpha, beta) on (0, 1), then applies an AffineTransform
    to shift to (-1, 1). Supports .mode, .mean, .entropy(), .log_prob().
    
    Constructor signature matches ProbabilisticActor's in_keys mapping:
        ScaledBeta(alpha=..., beta=...)
    """
    
    def __init__(self, alpha: torch.Tensor, beta: torch.Tensor):
        # Ensure valid parameters: alpha, beta > 1 for unimodal distribution
        self._alpha = alpha
        self._beta = beta
        base_dist = D.Beta(alpha, beta)
        # Transform from (0, 1) → (-1, 1): y = 2*x - 1
        transforms = [D.AffineTransform(loc=-1.0, scale=2.0)]
        super().__init__(base_dist, transforms)
    
    @property
    def mode(self):
        """Mode of the scaled Beta: 2 * mode_beta - 1."""
        alpha = self._alpha
        beta = self._beta
        # Beta mode = (alpha - 1) / (alpha + beta - 2), valid when alpha, beta > 1
        base_mode = (alpha - 1.0) / (alpha + beta - 2.0).clamp(min=1e-6)
        base_mode = base_mode.clamp(0.0, 1.0)
        return 2.0 * base_mode - 1.0
    
    @property
    def mean(self):
        """Mean of the scaled Beta: 2 * mean_beta - 1."""
        base_mean = self.base_dist.mean
        return 2.0 * base_mean - 1.0
    
    def entropy(self):
        """Analytic entropy of the scaled Beta.
        
        Entropy of affine-transformed distribution:
        H(Y) = H(X) + log|scale| = H(Beta) + log(2)
        """
        return self.base_dist.entropy() + np.log(2.0)
    
    def log_prob(self, value):
        """Log probability in the scaled space [-1, 1].
        
        log p(y) = log p_beta((y+1)/2) - log(2)
        """
        # Map from [-1, 1] back to (0, 1)
        x = (value + 1.0) / 2.0
        # Clamp to valid range for Beta log_prob
        x = x.clamp(1e-6, 1.0 - 1e-6)
        # Jacobian of the inverse transform: dx/dy = 1/2, so log|det| = -log(2)
        return self.base_dist.log_prob(x) - np.log(2.0)
    
    def rsample(self, sample_shape=torch.Size()):
        """Reparameterized sample in [-1, 1]."""
        x = self.base_dist.rsample(sample_shape)
        return 2.0 * x - 1.0
    
    def sample(self, sample_shape=torch.Size()):
        """Sample in [-1, 1]."""
        x = self.base_dist.sample(sample_shape)
        return 2.0 * x - 1.0
    
    @property
    def has_rsample(self):
        return self.base_dist.has_rsample


class IndependentScaledBeta(D.Distribution):
    """Wraps ScaledBeta to treat each action dimension as independent.
    
    ProbabilisticActor expects log_prob to return a scalar per batch element
    (summed over action dims). This wrapper handles that, plus provides
    .mode, .mean, and .deterministic_sample properties.
    """
    
    def __init__(self, alpha: torch.Tensor, beta: torch.Tensor):
        self._scaled_beta = ScaledBeta(alpha, beta)
        batch_shape = alpha.shape[:-1]
        event_shape = alpha.shape[-1:]
        super().__init__(batch_shape, event_shape, validate_args=False)
    
    @property
    def deterministic_sample(self):
        """Required by TorchRL ProbabilisticActor for DETERMINISTIC mode."""
        return self._scaled_beta.mode
    
    @property
    def mode(self):
        return self._scaled_beta.mode
    
    @property
    def mean(self):
        return self._scaled_beta.mean
    
    def entropy(self):
        # Sum over action dimensions
        return self._scaled_beta.entropy().sum(-1)
    
    def log_prob(self, value):
        # Sum over action dimensions
        return self._scaled_beta.log_prob(value).sum(-1)
    
    def rsample(self, sample_shape=torch.Size()):
        return self._scaled_beta.rsample(sample_shape)
    
    def sample(self, sample_shape=torch.Size()):
        return self._scaled_beta.sample(sample_shape)
    
    @property
    def has_rsample(self):
        return self._scaled_beta.has_rsample


# ============================================================
# Network Components
# ============================================================

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
        if x.dim() == 5:
            b, t, c, w, h = x.shape
            x = x.reshape(b * t, c, w, h)
            x = self.net(x)
            x = x.view(b, t, -1)
            return x
        elif x.dim() == 4 or x.dim() == 3:
            return self.net(x)
        else:
            raise RuntimeError(f"Unexpected lidar tensor shape: {tuple(x.shape)}")


class BetaResidualActionModule(nn.Module):
    """Residual module for Beta distribution.
    
    Operates in the [0, 1] space (Beta's natural domain).
    Adds a learned mean_delta to the human action's normalized position,
    then computes Beta parameters (alpha, beta) with a +1 offset that
    guarantees alpha, beta > 1 (unimodal distribution) for all inputs.
    
    Parameterization:
        alpha = mean * concentration + 1
        beta  = (1 - mean) * concentration + 1
    
    Properties:
        - Mode = mean  (exactly, always well-defined since alpha,beta > 1)
        - E[X] = (mean*conc + 1) / (conc + 2)  (approaches mean as conc → ∞)
        - Variance ≈ mean*(1-mean) / conc  (for large conc)
    
    When mean_delta ≈ 0, mode = human_action (identity initialization).
    """
    
    MAX_CONCENTRATION = 100.0  # Hard cap to prevent numerical instability
    
    def __init__(self, action_limit, min_concentration=2.0):
        super().__init__()
        self.action_limit = action_limit
        self.min_concentration = min_concentration
        self.register_buffer("residual_scale", torch.tensor(1.0))
    
    def forward(self, mean_delta, raw_concentration, human_action):
        # 1) Map human action from [-action_limit, +action_limit] to [0, 1]
        human_action_norm = human_action / self.action_limit
        human_action_01 = (human_action_norm + 1.0) / 2.0
        human_action_01 = human_action_01.clamp(0.01, 0.99)
        
        # 2) Apply residual in [0, 1] space
        mean = human_action_01 + mean_delta * self.residual_scale
        mean = mean.clamp(0.01, 0.99)
        
        # 3) Compute concentration (controls distribution width)
        #    Capped at MAX_CONCENTRATION to prevent numerical instability
        concentration = F.softplus(raw_concentration).clamp(max=self.MAX_CONCENTRATION) + self.min_concentration
        
        # 4) Parameterize Beta with +1 offset to guarantee alpha, beta > 1
        #    This ensures the distribution is always unimodal, and:
        #    - mode = (alpha-1)/(alpha+beta-2) = mean  (exact, always valid)
        #    - Deterministic eval uses mode → outputs exactly mean → clean residual
        alpha = mean * concentration + 1.0
        beta = (1.0 - mean) * concentration + 1.0
        
        return alpha, beta
    
    def set_scale(self, scale):
        self.residual_scale.fill_(scale)
    
    def set_min_concentration(self, value):
        """Adjust minimum concentration for curriculum control."""
        self.min_concentration = value


class DirectBetaActionModule(nn.Module):
    """Direct Beta action head for no-residual ablations.

    The policy still receives the pilot command in the observation encoder, but
    the Beta mean is not mathematically centered on that command.
    """

    MAX_CONCENTRATION = BetaResidualActionModule.MAX_CONCENTRATION

    def __init__(self, min_concentration=2.0):
        super().__init__()
        self.min_concentration = min_concentration

    def forward(self, mean_logits, raw_concentration):
        mean = torch.sigmoid(mean_logits).clamp(0.01, 0.99)
        concentration = F.softplus(raw_concentration).clamp(max=self.MAX_CONCENTRATION) + self.min_concentration
        alpha = mean * concentration + 1.0
        beta = (1.0 - mean) * concentration + 1.0
        return alpha, beta

    def set_min_concentration(self, value):
        self.min_concentration = value


class BetaSplitLayer(nn.Module):
    """Splits actor network output into mean_delta and raw_concentration."""
    
    def __init__(self, action_dim):
        super().__init__()
        self.action_dim = action_dim
    
    def forward(self, x):
        mean_delta, raw_concentration = x.split(self.action_dim, dim=-1)
        return mean_delta, raw_concentration


# ============================================================
# Main Algorithm
# ============================================================

class ConstrainedResidualPPO_Beta(TensorDictModuleBase):
    """Constrained Residual PPO with Beta distribution.
    
    API-compatible with ConstrainedResidualPPO (TanhNormal version).
    Key differences:
      - Uses IndependentScaledBeta distribution instead of TanhNormal
      - Residual in action space [0,1] instead of pre-tanh space
      - Analytic entropy instead of Monte Carlo estimate
      - No boundary clamp hacks needed
    """
    
    def __init__(self, cfg, observation_spec, action_spec, device):
        super().__init__()
        self.cfg = cfg
        self.device = device

        self.using_rnn = cfg.rnn.enable
        
        # Residual regularization coefficient
        self.reg_coeff = cfg.get("reg_coeff", 0.01)
        risk_reg_cfg = cfg.get("risk_regularization", {})
        self.use_risk_regularization = bool(risk_reg_cfg.get("enable", False))
        self.risk_reg_g_safe = float(risk_reg_cfg.get("g_safe", 1.0))
        self.risk_reg_g_danger = float(risk_reg_cfg.get("g_danger", 0.05))
        self.risk_reg_power = float(risk_reg_cfg.get("power", 1.0))
        self.policy_mode = cfg.get("policy_mode", "residual")
        if self.policy_mode not in ("residual", "direct"):
            raise ValueError("algo.policy_mode must be either 'residual' or 'direct'")
        
        # Beta-specific: minimum concentration (can be adjusted per curriculum stage)
        self.min_concentration = cfg.get("min_concentration", 2.0)

        # Get obs spec dims
        state_dim = observation_spec["agents", "observation", "state"].shape[-1]
        human_action_dim = observation_spec["agents", "observation", "human_action"].shape[-1]

        # Check if lidar is present
        self.has_lidar = "lidar" in observation_spec["agents", "observation"]
        self.has_critic_privileged = (
            "critic_privileged" in observation_spec["agents", "observation"]
        )
        
        modules = []
        cat_keys = []
        
        cnn_feature_dim = 0
        if self.has_lidar:
            feature_extractor_network = _LidarCNN().to(self.device)
            modules.append(TensorDictModule(feature_extractor_network, [("agents", "observation", "lidar")], ["_cnn_feature"]))
            cat_keys.append("_cnn_feature")
            cnn_feature_dim = 128

        cat_keys.extend([
            ("agents", "observation", "state"), 
            ("agents", "observation", "human_action"),
        ])

        if self.using_rnn:
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
            modules.append(CatTensors(
                in_keys=cat_keys, 
                out_key="_embed_inputs",
                del_keys=False
            ))
            modules.append(TensorDictModule(
                nn.LayerNorm(gru_input_dim, eps=NORM_EPS),
                in_keys=["_embed_inputs"],
                out_keys=["_embed"]
            ))
            modules.append(self.gru_model)
        else:
            modules.append(CatTensors(
                in_keys=cat_keys, 
                out_key="_embed_inputs",
                del_keys=False
            ))
            input_dim = cnn_feature_dim + state_dim + human_action_dim
            modules.append(TensorDictModule(
                nn.LayerNorm(input_dim, eps=NORM_EPS),
                in_keys=["_embed_inputs"],
                out_keys=["_embed"]
            ))
        
        modules.append(TensorDictModule(make_mlp([256, 256]), ["_embed"], ["_feature"]))
        self.feature_extractor = TensorDictSequential(*modules).to(self.device)

        # Actor network
        actual_action_spec = action_spec[("agents", "action")]
        self.n_agents, self.action_dim = actual_action_spec.shape[-2:]
        self.action_limit = cfg.actor.action_limit

        # Actor net outputs [mean_delta/mean_logits, raw_concentration], each of size action_dim
        self.actor_net = TensorDictModule(
            nn.Sequential(
                make_mlp([256, 256]), 
                nn.Linear(256, self.action_dim * 2)  # [mean_delta or mean_logits, raw_concentration]
            ),
            in_keys=["_feature"],
            out_keys=["_actor_logits"]
        ).to(self.device)

        # Split into mean_delta/mean_logits and raw_concentration
        split_module = TensorDictModule(
            BetaSplitLayer(self.action_dim),
            in_keys=["_actor_logits"],
            out_keys=["_mean_delta", "_raw_concentration"]
        )

        if self.policy_mode == "residual":
            # Residual module: combines mean_delta + human_action → alpha, beta
            self.action_parameter_module = BetaResidualActionModule(
                self.action_limit,
                min_concentration=self.min_concentration
            )
            beta_param_module = TensorDictModule(
                self.action_parameter_module,
                in_keys=["_mean_delta", "_raw_concentration", ("agents", "observation", "human_action")],
                out_keys=["alpha", "beta"]
            )
        else:
            self.action_parameter_module = DirectBetaActionModule(
                min_concentration=self.min_concentration
            )
            beta_param_module = TensorDictModule(
                self.action_parameter_module,
                in_keys=["_mean_delta", "_raw_concentration"],
                out_keys=["alpha", "beta"]
            )

        # ProbabilisticActor with IndependentScaledBeta
        self.actor = ProbabilisticActor(
            module=TensorDictSequential(self.actor_net, split_module, beta_param_module),
            in_keys=["alpha", "beta"],
            out_keys=[("agents", "action_normalized")],
            distribution_class=IndependentScaledBeta,
            return_log_prob=True,
            log_prob_key="sample_log_prob"
        ).to(self.device)

        self.critic_feature_extractor = None
        critic_in_key = "_feature"
        if self.has_critic_privileged:
            privileged_dim = observation_spec[
                "agents", "observation", "critic_privileged"
            ].shape[-1]
            critic_input_dim = 256 + privileged_dim
            self.critic_feature_extractor = TensorDictSequential(
                CatTensors(
                    in_keys=["_feature", ("agents", "observation", "critic_privileged")],
                    out_key="_critic_inputs",
                    del_keys=False,
                ),
                TensorDictModule(
                    nn.LayerNorm(critic_input_dim, eps=NORM_EPS),
                    in_keys=["_critic_inputs"],
                    out_keys=["_critic_inputs_norm"],
                ),
                TensorDictModule(
                    make_mlp([256]),
                    ["_critic_inputs_norm"],
                    ["_critic_feature"],
                ),
            ).to(self.device)
            critic_in_key = "_critic_feature"

        # Critic network (optional privileged branch; actor path stays unchanged)
        self.critic = TensorDictModule(
            nn.LazyLinear(1), [critic_in_key], ["state_value"]
        ).to(self.device)
        self.value_norm = ValueNorm(1).to(self.device)

        # Loss related
        self.gae = GAE(0.99, 0.95)
        self.critic_loss_fn = nn.HuberLoss(delta=10, reduction='none') 

        # Optimizer
        self.feature_extractor_optim = torch.optim.Adam(self.feature_extractor.parameters(), lr=cfg.feature_extractor.learning_rate)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor.learning_rate)
        critic_params = self.critic.parameters()
        if self.critic_feature_extractor is not None:
            critic_params = chain(self.critic_feature_extractor.parameters(), critic_params)
        self.critic_optim = torch.optim.Adam(critic_params, lr=cfg.critic.learning_rate)

        # Dummy Input for lazy modules
        dummy_input = observation_spec.zero()

        if self.using_rnn:
            dummy_input.set("is_init", torch.ones(dummy_input.batch_size, dtype=torch.bool, device=self.device))
            dummy_input.set(
                "recurrent_state",
                torch.zeros(
                    (*dummy_input.batch_size, self.gru_num_layers, self.gru_hidden_dim),
                    device=self.device,
                ),
            )

        self.__call__(dummy_input)

        # Initialize network weights
        def init_(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.)

        self.feature_extractor.apply(init_) 
        self.actor.apply(init_)
        if self.critic_feature_extractor is not None:
            self.critic_feature_extractor.apply(init_)
        self.critic.apply(init_)

        # Last-layer initialization:
        # residual mode: mean_delta ≈ 0, so deterministic action starts near human input.
        # direct mode: mean_logits ≈ 0, so deterministic action starts near zero velocity.
        def init_beta_output(module):
            if isinstance(module, nn.Linear):
                nn.init.constant_(module.weight, 1e-6)
                nn.init.constant_(module.bias, 0.)

        self.actor_net.module[-1].apply(init_beta_output)

    @staticmethod
    def _upgrade_legacy_state_dict_keys(state_dict):
        """Map old residual Beta action-module keys to the current module name."""
        legacy_prefix = "residual_action_module."
        current_prefix = "action_parameter_module."
        if not any(key.startswith(legacy_prefix) for key in state_dict):
            return state_dict

        upgraded = OrderedDict()
        for key, value in state_dict.items():
            if key.startswith(legacy_prefix):
                new_key = current_prefix + key[len(legacy_prefix):]
                if new_key not in state_dict:
                    upgraded[new_key] = value
                continue
            upgraded[key] = value

        if hasattr(state_dict, "_metadata"):
            upgraded._metadata = state_dict._metadata
        return upgraded

    def load_state_dict(self, state_dict, strict=True, assign=False):
        state_dict = self._upgrade_legacy_state_dict_keys(state_dict)
        try:
            return super().load_state_dict(state_dict, strict=strict, assign=assign)
        except TypeError:
            return super().load_state_dict(state_dict, strict=strict)
    
    def set_reg_coeff(self, value):
        """Set the residual regularization coefficient (for curriculum)."""
        self.reg_coeff = value

    def set_residual_scale(self, scale):
        """Set the scale of the residual policy output (0.0 to 1.0)."""
        if self.policy_mode == "residual" and hasattr(self, "action_parameter_module"):
            self.action_parameter_module.set_scale(scale)
    
    def set_min_concentration(self, value):
        """Adjust minimum concentration (higher → less exploration noise).
        
        Recommended values:
          - Early stages: 2.0 (wide exploration)
          - Mid stages: 5.0 (moderate)
          - Late stages: 10-20 (narrow, precise)
        """
        if hasattr(self, "action_parameter_module"):
            self.action_parameter_module.set_min_concentration(value)

    def __call__(self, tensordict):
        # Input validation (same as TanhNormal version)
        obs_state = tensordict.get(("agents", "observation", "state"), None)
        if obs_state is not None and torch.isnan(obs_state).any():
             print(f"[PPO-Beta Debug] NaN in State Input: {obs_state}")
             raise ValueError("NaN detected in Input: State")

        obs_human = tensordict.get(("agents", "observation", "human_action"), None)
        if obs_human is not None and torch.isnan(obs_human).any():
             print(f"[PPO-Beta Debug] NaN in Human Action Input")
             raise ValueError("NaN detected in Input: Human Action")

        if self.has_lidar:
             obs_lidar = tensordict.get(("agents", "observation", "lidar"), None)
             if obs_lidar is not None and torch.isnan(obs_lidar).any():
                 print(f"[PPO-Beta Debug] NaN in Lidar Input")
                 raise ValueError("NaN detected in Input: Lidar")

        self.feature_extractor(tensordict)
        self._apply_critic_features(tensordict)
        
        if torch.isnan(tensordict.get("_feature")).any():
            for name, param in self.feature_extractor.named_parameters():
                 if torch.isnan(param).any():
                      raise ValueError(f"NaN detected in Feature Extractor Weights: {name}")
            raise ValueError("NaN in Feature Extractor Output")

        self.actor(tensordict)
        self.critic(tensordict)

        # Beta samples are already in (-1, 1) → scale to action space
        action_norm = tensordict["agents", "action_normalized"]
        actions = action_norm * self.cfg.actor.action_limit

        # Transform to world frame
        actions_world = vec_to_world(
            actions, tensordict["agents", "observation", "state"], yaw_only=True
        )

        tensordict["agents", "action"] = actions_world
        tensordict["agents", "command"] = actions_world.clone() 
        return tensordict

    def get_recurrent_primer(self):
        if self.using_rnn:
            primer = self.gru_model.make_tensordict_primer()
            return primer
        return None

    def train_op(self, tensordict):
        profiler = get_profiler()
        
        with profiler.timer("ppo/compute_gae"):
            next_tensordict = tensordict["next"]
            with torch.no_grad():
                self.feature_extractor(next_tensordict)
                self._apply_critic_features(next_tensordict)
                next_values = self.critic(next_tensordict)["state_value"]
            rewards = tensordict["next", "agents", "reward"]
            dones = tensordict["next", "terminated"]

            values = tensordict["state_value"] 
            values = self.value_norm.denormalize(values)
            next_values = self.value_norm.denormalize(next_values)

            adv, ret = self.gae(rewards, dones, values, next_values)

            adv_mean = adv.mean()
            adv_std = adv.std()
            adv = (adv - adv_mean) / adv_std.clip(1e-7)

            self.value_norm.update(ret) 
            ret = self.value_norm.normalize(ret)  

            tensordict.set("adv", adv)
            tensordict.set("ret", ret)

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
            self._apply_critic_features(minibatch)
            
            # NaN guard: if feature extractor outputs NaN, skip this minibatch
            features = minibatch.get("_feature")
            if features is not None and torch.isnan(features).any():
                print("[PPO-Beta] WARNING: NaN in features during _update, skipping minibatch")
                return self._nan_info_dict()

            action_dist = self.actor.get_dist(minibatch) 

            action = minibatch[("agents", "action_normalized")]
            # Beta: actions in (-1, 1). Clamp slightly inward for numerical safety
            action_safe = action.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
            log_probs = action_dist.log_prob(action_safe)
            # log_prob is already summed over action dims by IndependentScaledBeta
            if log_probs.dim() > action_safe.dim() - 1:
                log_probs = log_probs.sum(dim=-1)

        with profiler.timer("ppo/update/loss_compute"):
            # 1. Entropy Loss — analytic, no Monte Carlo needed
            entropy = action_dist.entropy()
            entropy_loss = -self.cfg.entropy_loss_coefficient * torch.mean(entropy)

            # 2. Actor Loss (PPO clipped objective)
            advantage = minibatch["adv"] 
            old_log_probs = minibatch["sample_log_prob"]
            if old_log_probs.dim() > log_probs.dim():
                old_log_probs = old_log_probs.sum(dim=-1)
            # Clamp log-ratio before exp to prevent inf/NaN
            log_ratio = (log_probs - old_log_probs).clamp(-20.0, 20.0)
            ratio = torch.exp(log_ratio).unsqueeze(-1)
            ratio = ratio.clamp(max=10.0)
            surr1 = advantage * ratio
            surr2 = advantage * ratio.clamp(1.-self.cfg.actor.clip_ratio, 1.+self.cfg.actor.clip_ratio)
            actor_loss = -torch.mean(torch.min(surr1, surr2)) * self.action_dim 

            # 3. Regularization Loss — penalize deviation from human intent.
            human_action = minibatch[("agents", "observation", "human_action")]
            human_action_norm = human_action / self.action_limit
            human_action_01 = ((human_action_norm + 1.0) / 2.0).clamp(0.01, 0.99)
            reg_gate = torch.ones_like(human_action_01[..., 0])
            modal_residual = torch.zeros_like(reg_gate)
            if self.use_risk_regularization:
                mode_01 = ((action_dist.mode + 1.0) / 2.0).clamp(0.01, 0.99)
                modal_residual = (mode_01 - human_action_01).pow(2).sum(dim=-1)
                pilot_risk = minibatch.get(("next", "agents", "pilot_risk_dyn_post"), None)
                if pilot_risk is None:
                    raise KeyError("risk_regularization requires next/agents/pilot_risk_dyn_post")
                pilot_risk = pilot_risk.detach().squeeze(-1).clamp(0.0, 1.0)
                reg_gate = self.risk_reg_g_danger + (
                    self.risk_reg_g_safe - self.risk_reg_g_danger
                ) * (1.0 - pilot_risk).clamp(0.0, 1.0).pow(self.risk_reg_power)
                reg_loss = (reg_gate * modal_residual).mean()
            elif self.policy_mode == "residual":
                modal_residual = minibatch["_mean_delta"].pow(2).sum(dim=-1)
                reg_loss = modal_residual.mean()
            else:
                direct_mean_01 = torch.sigmoid(minibatch["_mean_delta"]).clamp(0.01, 0.99)
                modal_residual = (direct_mean_01 - human_action_01).pow(2).sum(dim=-1)
                reg_loss = modal_residual.mean()

            # 4. Policy Loss
            loss_pi = actor_loss + self.reg_coeff * reg_loss
            
            # 5. Critic Loss (identical to TanhNormal version)
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
            self.feature_extractor_optim.zero_grad()
            self.actor_optim.zero_grad()
            self.critic_optim.zero_grad()
            loss.backward()

            # NaN guard: skip optimizer step if any gradient is NaN
            if self._has_nan_gradients():
                print("[PPO-Beta] WARNING: NaN gradients detected, skipping optimizer step")
                self.feature_extractor_optim.zero_grad()
                self.actor_optim.zero_grad()
                self.critic_optim.zero_grad()
                return self._nan_info_dict()

            feature_extractor_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.feature_extractor.parameters(), max_norm=5.)
            actor_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.actor.parameters(), max_norm=5.) 
            critic_params_for_clip = self.critic.parameters()
            if self.critic_feature_extractor is not None:
                critic_grad_norm = nn.utils.clip_grad.clip_grad_norm_(
                    chain(self.critic_feature_extractor.parameters(), critic_params_for_clip),
                    max_norm=5.,
                )
            else:
                critic_grad_norm = nn.utils.clip_grad.clip_grad_norm_(
                    critic_params_for_clip, max_norm=5.
                )
            
            self.feature_extractor_optim.step()
            self.actor_optim.step()
            self.critic_optim.step()

            explained_var = 1 - F.mse_loss(value, ret) / ret.var()
            
            return TensorDict({
                "actor_loss": actor_loss,
                "critic_loss": critic_loss,
                "entropy": entropy_loss,
                "reg_loss": reg_loss,
                "reg_gate": reg_gate.mean(),
                "modal_residual": modal_residual.mean(),
                "actor_grad_norm": actor_grad_norm,
                "critic_grad_norm": critic_grad_norm,
                "explained_var": explained_var
            }, [])
    
    def _has_nan_gradients(self):
        """Check if any parameter gradient contains NaN."""
        param_groups = [
            self.feature_extractor.parameters(),
            self.actor.parameters(),
            self.critic.parameters(),
        ]
        if self.critic_feature_extractor is not None:
            param_groups.append(self.critic_feature_extractor.parameters())
        for param_group in param_groups:
            for p in param_group:
                if p.grad is not None and torch.isnan(p.grad).any():
                    return True
        return False
    
    def _nan_info_dict(self):
        """Return a TensorDict with NaN values for logging when update is skipped."""
        return TensorDict({
            "actor_loss": torch.tensor(float('nan')),
            "critic_loss": torch.tensor(float('nan')),
            "entropy": torch.tensor(float('nan')),
            "reg_loss": torch.tensor(float('nan')),
            "reg_gate": torch.tensor(float('nan')),
            "modal_residual": torch.tensor(float('nan')),
            "actor_grad_norm": torch.tensor(float('nan')),
            "critic_grad_norm": torch.tensor(float('nan')),
            "explained_var": torch.tensor(float('nan')),
        }, [])

    def _apply_critic_features(self, tensordict):
        if self.critic_feature_extractor is not None:
            self.critic_feature_extractor(tensordict)

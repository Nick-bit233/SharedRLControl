"""
Standalone ConstrainedResidualPPO network for inference only.
No TorchRL / TensorDict dependency — pure PyTorch.

Architecture (training-identical, Beta distribution):
  LidarCNN: (N, 1, 36, 4) → 128
  Concat:   [cnn_feat(128), state(10), human_action(3)] → 141
  LayerNorm(141)
  MLP:      141 → 256 → 256  (_feature)
  ActorMLP: 256 → 256 → 256 → 6  (mean_delta[3], raw_concentration[3])
  BetaResidual:
    ha_01 = (ha / action_limit + 1) / 2              map to [0, 1]
    mean  = clamp(ha_01 + mean_delta * residual_scale, 0.01, 0.99)
    conc  = softplus(raw_concentration) + min_concentration
    alpha = mean * conc + 1
    beta  = (1 - mean) * conc + 1
    mode  = mean  (since alpha, beta > 1)
  Deterministic: action_norm = 2 * mode - 1           → [-1, 1]
  Scale:    action = action_norm * action_limit        → m/s
  Transform: body-frame → world-frame (yaw-only rotation)

Checkpoint: curriculum_stage3/checkpoint_final.pt (Beta distribution)
  Training config: distribution=beta, min_concentration=2.0, action_limit=2.0
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .quat_utils import vec_to_world

NORM_EPS = 1e-3


class _LidarCNN(nn.Module):
    """3-layer Conv2d for LiDAR range images. Input: (N, 1, 36, 4) → Output: (N, 128)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=(5, 3), padding=(2, 1)),
            nn.ELU(),
            nn.Conv2d(4, 16, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1)),
            nn.ELU(),
            nn.Conv2d(16, 16, kernel_size=(5, 3), stride=(2, 2), padding=(2, 1)),
            nn.ELU(),
            nn.Flatten(),  # (N, 16*9*2) = (N, 288)
            nn.Linear(288, 128),
            nn.LayerNorm(128, eps=NORM_EPS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _make_mlp(dims: list) -> nn.Sequential:
    layers = []
    for d in dims:
        layers.extend([nn.LazyLinear(d), nn.LeakyReLU(), nn.LayerNorm(d, eps=NORM_EPS)])
    return nn.Sequential(*layers)


class TunnelPolicyNet(nn.Module):
    """
    Self-contained inference network that mirrors the training-time
    ConstrainedResidualPPO_Beta architecture (Beta distribution).

    Call signature:
        action_world = net(state, human_action, lidar, deterministic=True)

    Where:
        state:        (N, 10)  [vel_b(3), ang_vel_b(3), quat(4)]
        human_action: (N, 3)   body-frame velocity command
        lidar:        (N, 1, 36, 4) normalised range image
    Returns:
        action_world: (N, 3)   world-frame velocity command  (m/s)
    """

    MAX_CONCENTRATION = 100.0

    def __init__(self, action_limit: float = 2.0, min_concentration: float = 2.0):
        super().__init__()
        self.action_limit = action_limit
        self.min_concentration = min_concentration

        # ---- Feature extractor (identical to TanhNormal version) ----
        self.lidar_cnn = _LidarCNN()
        self.input_norm = nn.LayerNorm(141, eps=NORM_EPS)
        self.feature_mlp = _make_mlp([256, 256])

        # ---- Actor head ----
        self.actor_mlp = nn.Sequential(
            _make_mlp([256, 256]),
            nn.Linear(256, 6),  # mean_delta(3) + raw_concentration(3)
        )

        # ---- Residual module ----
        self.residual_scale = nn.Parameter(torch.tensor(1.0), requires_grad=False)

    # ------------------------------------------------------------------
    # Forward (inference) — Beta distribution
    # ------------------------------------------------------------------
    def forward(
        self,
        state: torch.Tensor,
        human_action: torch.Tensor,
        lidar: torch.Tensor,
        deterministic: bool = True,
    ) -> torch.Tensor:
        # 1. Feature extraction
        cnn_feat = self.lidar_cnn(lidar)                     # (N, 128)
        cat_feat = torch.cat([cnn_feat, state, human_action], dim=-1)  # (N, 141)
        normed = self.input_norm(cat_feat)
        feature = self.feature_mlp(normed)                   # (N, 256)

        # 2. Actor outputs mean_delta and raw_concentration
        logits = self.actor_mlp(feature)                     # (N, 6)
        mean_delta, raw_concentration = logits.split(3, dim=-1)

        # 3. Beta residual in [0, 1] space
        #    (mirrors BetaResidualActionModule.forward from ppo_constrained_beta.py)
        ha_norm = human_action / self.action_limit            # [-1, 1]
        ha_01 = (ha_norm + 1.0) / 2.0                        # [0, 1]
        ha_01 = ha_01.clamp(0.01, 0.99)

        mean = ha_01 + mean_delta * self.residual_scale
        mean = mean.clamp(0.01, 0.99)

        concentration = F.softplus(raw_concentration).clamp(max=self.MAX_CONCENTRATION) + self.min_concentration
        alpha = mean * concentration + 1.0
        beta_ = (1.0 - mean) * concentration + 1.0

        if deterministic:
            # Mode of Beta(alpha, beta) = (alpha-1)/(alpha+beta-2) = mean
            action_norm = 2.0 * mean - 1.0                   # [-1, 1]
        else:
            dist = torch.distributions.Beta(alpha, beta_)
            sample_01 = dist.rsample()
            action_norm = 2.0 * sample_01 - 1.0              # [-1, 1]

        # 4. Scale to physical units
        action_body = action_norm * self.action_limit         # (N, 3) body-frame m/s

        # 5. Transform body → world (yaw-only)
        action_world = vec_to_world(action_body, state, yaw_only=True)

        # Debug: log internals for first N calls
        if not hasattr(self, '_fwd_count'):
            self._fwd_count = 0
        self._fwd_count += 1
        if self._fwd_count <= 15 or self._fwd_count % 200 == 0:
            md = mean_delta.squeeze(0).detach().cpu().numpy()
            m = mean.squeeze(0).detach().cpu().numpy()
            ab = action_body.squeeze(0).detach().cpu().numpy()
            aw = action_world.squeeze(0).detach().cpu().numpy()
            # Lidar: forward bins (0,1,34,35), left bins (8,9,10), right (26,27,28)
            L = lidar.squeeze().detach().cpu().numpy()  # (36, 4)
            fwd_mean = L[[0,1,34,35], :].mean()
            left_mean = L[[8,9,10], :].mean()
            right_mean = L[[26,27,28], :].mean()
            back_mean = L[[17,18,19], :].mean()
            print(f"[PolicyDbg] #{self._fwd_count} "
                  f"md=[{md[0]:.3f},{md[1]:.3f},{md[2]:.3f}] "
                  f"mean=[{m[0]:.3f},{m[1]:.3f},{m[2]:.3f}] "
                  f"body=[{ab[0]:.2f},{ab[1]:.2f},{ab[2]:.2f}] "
                  f"world=[{aw[0]:.2f},{aw[1]:.2f},{aw[2]:.2f}] "
                  f"lidar F={fwd_mean:.2f} L={left_mean:.2f} R={right_mean:.2f} B={back_mean:.2f}")
            # Dump exact tensors for offline replay
            if self._fwd_count <= 5:
                try:
                    dump = {
                        'state': state.detach().cpu(),
                        'human_action': human_action.detach().cpu(),
                        'lidar': lidar.detach().cpu(),
                        'md': mean_delta.detach().cpu(),
                        'mean': mean.detach().cpu(),
                        'body': action_body.detach().cpu(),
                        'world': action_world.detach().cpu(),
                    }
                    torch.save(dump, f'/tmp/policy_dump_{self._fwd_count}.pt')
                    print(f"[PolicyDbg] Dumped tensors to /tmp/policy_dump_{self._fwd_count}.pt")
                except Exception as e:
                    print(f"[PolicyDbg] Dump error: {e}")

        return action_world

    # ------------------------------------------------------------------
    # Checkpoint loading
    # ------------------------------------------------------------------
    @classmethod
    def from_checkpoint(cls, ckpt_path: str, action_limit: float = 2.0,
                        min_concentration: float = 2.0,
                        device: str = "cpu") -> "TunnelPolicyNet":
        """
        Load a training checkpoint and map weights into this standalone net.

        The training checkpoint uses TensorDict module paths like:
            feature_extractor.module.0.module.net.0.weight   (LidarCNN)
            feature_extractor.module.2.module.{weight|bias}  (LayerNorm)
            feature_extractor.module.3.module.{idx}.*        (feature MLP)
            actor_net.module.0.{idx}.*                       (actor MLP layers)
            actor_net.module.1.*                              (final Linear 256→6)
            residual_action_module.residual_scale             (scalar)
        """
        net = cls(action_limit=action_limit,
                  min_concentration=min_concentration).to(device)

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "policy" in ckpt:
            src = ckpt["policy"]
        elif isinstance(ckpt, dict) and any(k.startswith("feature_extractor") for k in ckpt):
            src = ckpt
        else:
            src = ckpt

        mapped = _map_checkpoint(src, net, device)

        missing, unexpected = net.load_state_dict(mapped, strict=False)
        if missing:
            print(f"[TunnelPolicyNet] Missing keys (expected for critic): {missing}")
        if unexpected:
            print(f"[TunnelPolicyNet] Unexpected keys: {unexpected}")

        # Materialise any remaining lazy layers with a dummy forward
        dummy_state = torch.zeros(1, 10, device=device)
        dummy_ha = torch.zeros(1, 3, device=device)
        dummy_lidar = torch.zeros(1, 1, 36, 4, device=device)
        with torch.no_grad():
            net(dummy_state, dummy_ha, dummy_lidar)

        net.eval()
        return net


# ======================================================================
# Weight mapping helpers
# ======================================================================
def _map_checkpoint(src: dict, net: "TunnelPolicyNet", device: str) -> dict:
    """
    Heuristic mapper: walk the source state_dict and figure out where
    each tensor belongs in the standalone network.
    """
    dst = {}

    for k, v in src.items():
        target_key = _translate_key(k)
        if target_key is not None:
            dst[target_key] = v.to(device)

    return dst


def _translate_key(k: str):
    """Map a single training-checkpoint key to the standalone key, or None to skip.

    Checkpoint has duplicated paths (same tensor via different module refs).
    We prefer the shorter ``actor_net.module.*`` path over the longer
    ``actor.module.0.module.0.module.*`` path.
    """

    # ---- LiDAR CNN ----
    # Training:  feature_extractor.module.0.module.net.{idx}.{weight|bias}
    # Standalone: lidar_cnn.net.{idx}.{weight|bias}
    prefix_cnn = "feature_extractor.module.0.module.net."
    if k.startswith(prefix_cnn):
        suffix = k[len(prefix_cnn):]
        return f"lidar_cnn.net.{suffix}"

    # ---- Input LayerNorm ----
    # Training:  feature_extractor.module.2.module.{weight|bias}
    prefix_inorm = "feature_extractor.module.2.module."
    if k.startswith(prefix_inorm):
        suffix = k[len(prefix_inorm):]
        return f"input_norm.{suffix}"

    # ---- Feature MLP ----
    # Training:  feature_extractor.module.3.module.{idx}.{weight|bias}
    prefix_fmlp = "feature_extractor.module.3.module."
    if k.startswith(prefix_fmlp):
        suffix = k[len(prefix_fmlp):]
        return f"feature_mlp.{suffix}"

    # ---- Actor MLP (short path via self.actor_net) ----
    # Training:  actor_net.module.0.{idx}.{weight|bias}   (make_mlp layers)
    #            actor_net.module.1.{weight|bias}          (final Linear(256→6))
    # Standalone: actor_mlp.0.{idx}.{weight|bias}
    #             actor_mlp.1.{weight|bias}
    prefix_actornet = "actor_net.module."
    if k.startswith(prefix_actornet):
        suffix = k[len(prefix_actornet):]
        return f"actor_mlp.{suffix}"

    # ---- Residual scale (short path) ----
    if k == "residual_action_module.residual_scale":
        return "residual_scale"

    # Skip: critic, value_norm, gae, duplicate actor.module.* paths,
    #        and the duplicate actor.module.0.module.2.module.residual_scale
    return None

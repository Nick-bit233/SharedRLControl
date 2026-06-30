"""Real-room SRLC training environment.

This class reuses the experiment-04 tunnel environment implementation with
`env.name=real_room`, which activates the small-room geometry, fixed obstacles,
and room-specific termination/reward settings.
"""

from src.envs.env_tunnel import EnvTunnelResidual


class EnvRealRoomResidual(EnvTunnelResidual):
    pass

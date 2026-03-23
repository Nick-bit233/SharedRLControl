"""
Curriculum scheduler for Phase 2: reg_coeff ramp.

Tracks an EMA of success_rate and increases reg_coeff when the metric
stays above a promotion threshold for `patience` consecutive checks.
"""


class RegCoeffScheduler:
    """Gradually increase reg_coeff based on smoothed success_rate."""

    def __init__(self, cfg):
        self.check_interval = cfg.check_interval
        self.ema_alpha = cfg.ema_alpha
        self.promotion_threshold = cfg.promotion_threshold
        self.demotion_threshold = cfg.demotion_threshold
        self.patience = cfg.patience
        self.reg_step = cfg.reg_step
        self.max_reg_coeff = cfg.max_reg_coeff

        self.current_reg_coeff = cfg.initial_reg_coeff
        self.ema_success = 0.0
        self._above_count = 0
        self._initialized = False

    def update(self, success_rate: float) -> float:
        """Feed a new success_rate observation. Returns current reg_coeff."""
        if not self._initialized:
            self.ema_success = success_rate
            self._initialized = True
        else:
            self.ema_success = (
                self.ema_alpha * success_rate
                + (1 - self.ema_alpha) * self.ema_success
            )

        if self.ema_success >= self.promotion_threshold:
            self._above_count += 1
        else:
            self._above_count = 0

        if self._above_count >= self.patience:
            self.current_reg_coeff = min(
                self.current_reg_coeff + self.reg_step,
                self.max_reg_coeff,
            )
            self._above_count = 0

        return self.current_reg_coeff

    def state_dict(self):
        return {
            "current_reg_coeff": self.current_reg_coeff,
            "ema_success": self.ema_success,
            "_above_count": self._above_count,
            "_initialized": self._initialized,
        }

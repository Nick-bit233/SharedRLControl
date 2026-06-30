"""
Curriculum scheduler: only for reg_coeff ramp.

Tracks an EMA of success_rate and increases reg_coeff when the metric
stays above a promotion threshold for `patience` consecutive checks.
"""

from __future__ import annotations

from typing import Any


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

    def load_state_dict(self, state):
        self.current_reg_coeff = state.get("current_reg_coeff", self.current_reg_coeff)
        self.ema_success = state.get("ema_success", self.ema_success)
        self._above_count = state.get("_above_count", self._above_count)
        self._initialized = state.get("_initialized", self._initialized)


class RegCoeffSchedulerHook:
    """Optional lifecycle hook for experiments that use residual reg_coeff curriculum."""

    def __init__(
        self,
        *,
        cfg_key: str = "curriculum",
        checkpoint_key: str = "reg_scheduler",
        metric_context_key: str = "latest_eval_success",
        log_prefix: str = "curriculum",
        ema_metric_name: str = "ema_success",
    ) -> None:
        self.cfg_key = cfg_key
        self.checkpoint_key = checkpoint_key
        self.metric_context_key = metric_context_key
        self.log_prefix = log_prefix
        self.ema_metric_name = ema_metric_name
        self.scheduler: RegCoeffScheduler | None = None

    def on_after_setup(self, context: dict[str, Any]) -> None:
        cfg = context["cfg"]
        curriculum_cfg = cfg.get(self.cfg_key, {})
        if not curriculum_cfg.get("enable", False):
            return

        self.scheduler = RegCoeffScheduler(curriculum_cfg)
        print(
            f"[Train] Curriculum ENABLED: reg_coeff will ramp from "
            f"{curriculum_cfg.initial_reg_coeff} to {curriculum_cfg.max_reg_coeff}"
        )

        resume_state = context["checkpoint_state"].resume_state
        if resume_state is not None and self.checkpoint_key in resume_state:
            try:
                self.scheduler.load_state_dict(resume_state[self.checkpoint_key])
                print(
                    f"[Train] RegCoeffScheduler state restored: "
                    f"reg_coeff={self.scheduler.current_reg_coeff:.4f}, "
                    f"ema_success={self.scheduler.ema_success:.3f}"
                )
            except Exception as exc:
                print(f"[Train] WARNING: curriculum restore failed: {exc}")

        context["policy"].set_reg_coeff(self.scheduler.current_reg_coeff)
        self._store_checkpoint_state(context)

    def on_after_train_step(self, context: dict[str, Any]) -> None:
        if self.scheduler is None:
            return

        loop_iter = int(context.get("loop_iter", context.get("global_iter", 0)))
        if loop_iter % self.scheduler.check_interval != 0:
            return

        metric_value = context.get(self.metric_context_key)
        if metric_value is None:
            return

        new_reg = self.scheduler.update(float(metric_value))
        context["policy"].set_reg_coeff(new_reg)
        info = context.setdefault("info", {})
        info[f"{self.log_prefix}/reg_coeff"] = new_reg
        info[f"{self.log_prefix}/{self.ema_metric_name}"] = self.scheduler.ema_success
        self._store_checkpoint_state(context)

    def on_before_checkpoint(self, context: dict[str, Any]) -> None:
        self._store_checkpoint_state(context)

    def _store_checkpoint_state(self, context: dict[str, Any]) -> None:
        if self.scheduler is None:
            return
        context.setdefault("checkpoint_extra_state", {})[self.checkpoint_key] = self.scheduler.state_dict()


__all__ = [
    "RegCoeffScheduler",
    "RegCoeffSchedulerHook",
]

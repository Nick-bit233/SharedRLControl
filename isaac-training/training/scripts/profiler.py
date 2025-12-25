"""
Training Performance Profiler for RL Pipeline

This module provides timing utilities for analyzing training performance bottlenecks.
Supports:
- Nested timing contexts
- Automatic CUDA synchronization (optional, for accurate GPU timing)
- Statistical aggregation (mean, std, total, count)
- Wandb integration for visualization
- Logging to file

Usage:
    from profiler import get_profiler
    
    profiler = get_profiler()
    
    with profiler.timer("data_collection"):
        # ... data collection code ...
        with profiler.timer("env_step"):
            # ... env step code ...
    
    # At the end of training loop:
    profiler.log_to_wandb(wandb_run)
    profiler.print_summary()
"""

import time
import torch
import logging
from typing import Dict, Optional, List
from contextlib import contextmanager
from collections import defaultdict
import numpy as np


class TimingStats:
    """Statistics for a single timing metric."""
    
    def __init__(self):
        self.times: List[float] = []
        self.total_time: float = 0.0
        self.call_count: int = 0
    
    def add(self, elapsed: float):
        self.times.append(elapsed)
        self.total_time += elapsed
        self.call_count += 1
    
    def get_stats(self) -> Dict[str, float]:
        if self.call_count == 0:
            return {"mean": 0.0, "std": 0.0, "total": 0.0, "count": 0, "percent": 0.0}
        
        times_array = np.array(self.times[-1000:])  # Keep last 1000 for memory efficiency
        return {
            "mean": float(np.mean(times_array)),
            "std": float(np.std(times_array)),
            "total": self.total_time,
            "count": self.call_count,
            "min": float(np.min(times_array)),
            "max": float(np.max(times_array)),
        }
    
    def reset(self):
        self.times.clear()
        self.total_time = 0.0
        self.call_count = 0


class TrainingProfiler:
    """
    Hierarchical timing profiler for RL training pipeline.
    
    Features:
    - Nested timing contexts with automatic hierarchy tracking
    - CUDA synchronization for accurate GPU timing
    - Aggregated statistics per timing key
    - Wandb and logging integration
    """
    
    _instance: Optional["TrainingProfiler"] = None
    
    def __init__(
        self,
        enabled: bool = True,
        cuda_sync: bool = True,
        device: str = "cuda:0",
        log_file: Optional[str] = None,
    ):
        """
        Initialize the profiler.
        
        Args:
            enabled: If False, all timing operations become no-ops
            cuda_sync: If True, call torch.cuda.synchronize() before timing (accurate GPU timing)
            device: CUDA device for synchronization
            log_file: Optional file path for logging timing data
        """
        self.enabled = enabled
        self.cuda_sync = cuda_sync
        self.device = device
        
        self._timings: Dict[str, TimingStats] = defaultdict(TimingStats)
        self._stack: List[str] = []  # Current timing context stack
        self._start_times: Dict[str, float] = {}  # Active timers
        
        # Setup logging
        self.logger = logging.getLogger("TrainingProfiler")
        self.logger.setLevel(logging.DEBUG if enabled else logging.WARNING)
        
        # File handler if log_file specified
        if log_file:
            fh = logging.FileHandler(log_file)
            fh.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(ch)
        
        self._batch_count = 0
        self._last_log_batch = 0
    
    @classmethod
    def get_instance(cls, **kwargs) -> "TrainingProfiler":
        """Get or create the singleton profiler instance."""
        if cls._instance is None:
            cls._instance = TrainingProfiler(**kwargs)
        return cls._instance
    
    @classmethod
    def reset_instance(cls):
        """Reset the singleton instance."""
        cls._instance = None
    
    def _sync_cuda(self):
        """Synchronize CUDA if enabled."""
        if self.cuda_sync and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
    
    def _get_full_key(self, key: str) -> str:
        """Get hierarchical key based on current stack."""
        if self._stack:
            return "/".join(self._stack) + "/" + key
        return key
    
    @contextmanager
    def timer(self, key: str):
        """
        Context manager for timing a code block.
        
        Usage:
            with profiler.timer("my_operation"):
                # ... code to time ...
        """
        if not self.enabled:
            yield
            return
        
        full_key = self._get_full_key(key)
        self._stack.append(key)
        
        self._sync_cuda()
        start_time = time.perf_counter()
        
        try:
            yield
        finally:
            self._sync_cuda()
            elapsed = time.perf_counter() - start_time
            
            self._timings[full_key].add(elapsed)
            self._stack.pop()
    
    def start(self, key: str):
        """Start a named timer (for non-context-manager usage)."""
        if not self.enabled:
            return
        
        full_key = self._get_full_key(key)
        self._sync_cuda()
        self._start_times[full_key] = time.perf_counter()
    
    def stop(self, key: str):
        """Stop a named timer and record elapsed time."""
        if not self.enabled:
            return
        
        full_key = self._get_full_key(key)
        if full_key not in self._start_times:
            self.logger.warning(f"Timer '{full_key}' was not started!")
            return
        
        self._sync_cuda()
        elapsed = time.perf_counter() - self._start_times[full_key]
        self._timings[full_key].add(elapsed)
        del self._start_times[full_key]
    
    def record(self, key: str, elapsed: float):
        """Manually record a timing value."""
        if not self.enabled:
            return
        full_key = self._get_full_key(key)
        self._timings[full_key].add(elapsed)
    
    def increment_batch(self):
        """Increment batch counter."""
        self._batch_count += 1
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get all timing statistics."""
        return {key: stats.get_stats() for key, stats in self._timings.items()}
    
    def get_summary_dict(self, prefix: str = "profiler/") -> Dict[str, float]:
        """
        Get a flat dictionary suitable for wandb logging.
        
        Returns:
            Dictionary with keys like "profiler/env_step/mean", "profiler/env_step/total", etc.
        """
        result = {}
        all_stats = self.get_stats()
        
        # Calculate total time for percentage calculation
        root_total = sum(
            s["total"] for k, s in all_stats.items() 
            if "/" not in k  # Only top-level timers
        )
        
        for key, stats in all_stats.items():
            for stat_name, value in stats.items():
                result[f"{prefix}{key}/{stat_name}"] = value
            
            # Add percentage of total
            if root_total > 0:
                result[f"{prefix}{key}/percent"] = (stats["total"] / root_total) * 100
        
        return result
    
    def log_to_wandb(self, wandb_run, prefix: str = "profiler/"):
        """Log timing statistics to wandb."""
        if not self.enabled:
            return
        
        summary = self.get_summary_dict(prefix)
        wandb_run.log(summary)
    
    def print_summary(self, top_n: int = 20):
        """Print a summary of timing statistics to console."""
        if not self.enabled:
            return
        
        all_stats = self.get_stats()
        if not all_stats:
            self.logger.info("No timing data collected.")
            return
        
        # Sort by total time
        sorted_stats = sorted(
            all_stats.items(), 
            key=lambda x: x[1]["total"], 
            reverse=True
        )[:top_n]
        
        # Calculate total for percentage
        root_total = sum(
            s["total"] for k, s in all_stats.items() 
            if "/" not in k
        )
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("TRAINING PROFILER SUMMARY")
        self.logger.info("=" * 80)
        self.logger.info(f"{'Key':<45} {'Mean (ms)':<12} {'Total (s)':<12} {'Count':<10} {'%':<8}")
        self.logger.info("-" * 80)
        
        for key, stats in sorted_stats:
            mean_ms = stats["mean"] * 1000
            total_s = stats["total"]
            count = stats["count"]
            pct = (stats["total"] / root_total * 100) if root_total > 0 else 0
            
            # Indent nested keys
            indent = "  " * key.count("/")
            display_key = indent + key.split("/")[-1]
            
            self.logger.info(f"{display_key:<45} {mean_ms:<12.3f} {total_s:<12.3f} {count:<10} {pct:<8.1f}")
        
        self.logger.info("=" * 80)
        self.logger.info(f"Total profiled time: {root_total:.2f}s over {self._batch_count} batches")
        self.logger.info("=" * 80 + "\n")
    
    def reset(self):
        """Reset all timing statistics."""
        self._timings.clear()
        self._stack.clear()
        self._start_times.clear()
        self._batch_count = 0
        self._last_log_batch = 0


# Global profiler instance getter
def get_profiler(**kwargs) -> TrainingProfiler:
    """
    Get the global profiler instance.
    
    First call initializes with provided kwargs.
    Subsequent calls ignore kwargs and return existing instance.
    """
    return TrainingProfiler.get_instance(**kwargs)


def reset_profiler():
    """Reset the global profiler instance."""
    TrainingProfiler.reset_instance()


# Decorator for timing functions
def profile(key: Optional[str] = None):
    """
    Decorator to profile a function.
    
    Usage:
        @profile("my_function")
        def my_function():
            ...
        
        # Or auto-name from function:
        @profile()
        def my_function():
            ...
    """
    def decorator(func):
        nonlocal key
        if key is None:
            key = func.__name__
        
        def wrapper(*args, **kwargs):
            profiler = get_profiler()
            with profiler.timer(key):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator

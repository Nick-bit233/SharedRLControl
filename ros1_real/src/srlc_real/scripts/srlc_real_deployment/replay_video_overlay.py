"""Deterministic transparent HUD frames for flight-replay video composition."""

from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence, Tuple

from .flight_replay import FlightTimeline


Color = Tuple[int, int, int, int]
StickInput = Tuple[float, float, float]


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(float(value), lower), upper)


def replay_elapsed_for_video_time(
    video_time: float,
    *,
    lead_seconds: float,
    speed: float,
    replay_duration: float,
) -> float:
    """Map an encoded-video timestamp to the source replay timeline."""

    return _clamp(
        (float(video_time) - float(lead_seconds)) * float(speed),
        0.0,
        float(replay_duration),
    )


class StickHudRenderer:
    """Draw a compact right-stick trail and current command into RGBA bytes."""

    def __init__(self, width: int = 300, height: int = 260):
        if width < 180 or height < 180:
            raise ValueError("stick HUD must be at least 180x180")
        self.width = int(width)
        self.height = int(height)
        self.center = (self.width // 2, int(self.height * 0.59))
        self.radius = int(min(self.width * 0.32, self.height * 0.34))
        background = bytes((10, 16, 28, 174))
        self._static = bytearray(background * (self.width * self.height))
        self._draw_static()

    def _pixel(self, image: bytearray, x: int, y: int, color: Color) -> None:
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return
        offset = 4 * (y * self.width + x)
        image[offset : offset + 4] = bytes(color)

    def _disc(
        self,
        image: bytearray,
        x: int,
        y: int,
        radius: int,
        color: Color,
    ) -> None:
        radius_squared = radius * radius
        for offset_y in range(-radius, radius + 1):
            span = int(math.sqrt(max(0, radius_squared - offset_y * offset_y)))
            for offset_x in range(-span, span + 1):
                self._pixel(image, x + offset_x, y + offset_y, color)

    def _line(
        self,
        image: bytearray,
        start: Tuple[int, int],
        end: Tuple[int, int],
        color: Color,
        width: int = 1,
    ) -> None:
        x0, y0 = start
        x1, y1 = end
        dx = x1 - x0
        dy = y1 - y0
        steps = max(abs(dx), abs(dy), 1)
        radius = max(0, int(width) // 2)
        for step in range(steps + 1):
            alpha = step / float(steps)
            x = int(round(x0 + alpha * dx))
            y = int(round(y0 + alpha * dy))
            if radius:
                self._disc(image, x, y, radius, color)
            else:
                self._pixel(image, x, y, color)

    def _circle(
        self,
        image: bytearray,
        center: Tuple[int, int],
        radius: int,
        color: Color,
        width: int = 1,
    ) -> None:
        circumference = max(48, int(2.0 * math.pi * radius * 1.5))
        previous = None
        for index in range(circumference + 1):
            angle = 2.0 * math.pi * index / circumference
            point = (
                int(round(center[0] + radius * math.cos(angle))),
                int(round(center[1] + radius * math.sin(angle))),
            )
            if previous is not None:
                self._line(image, previous, point, color, width=width)
            previous = point

    def _draw_static(self) -> None:
        cx, cy = self.center
        self._circle(self._static, self.center, self.radius, (220, 230, 245, 185), 2)
        self._circle(
            self._static,
            self.center,
            self.radius // 2,
            (155, 170, 195, 90),
            1,
        )
        self._line(
            self._static,
            (cx - self.radius, cy),
            (cx + self.radius, cy),
            (180, 195, 215, 105),
            1,
        )
        self._line(
            self._static,
            (cx, cy - self.radius),
            (cx, cy + self.radius),
            (180, 195, 215, 105),
            1,
        )
        self._disc(self._static, cx, cy, 3, (235, 240, 250, 210))

        # Forward arrow at the top of the command circle.
        tip_y = cy - self.radius - 12
        self._line(
            self._static,
            (cx, tip_y + 10),
            (cx, tip_y),
            (220, 230, 245, 190),
            2,
        )
        self._line(
            self._static,
            (cx, tip_y),
            (cx - 5, tip_y + 6),
            (220, 230, 245, 190),
            2,
        )
        self._line(
            self._static,
            (cx, tip_y),
            (cx + 5, tip_y + 6),
            (220, 230, 245, 190),
            2,
        )

    def _point(self, value: Sequence[float]) -> Tuple[int, int]:
        forward = _clamp(value[0], -1.0, 1.0)
        lateral = _clamp(value[1], -1.0, 1.0)
        return (
            int(round(self.center[0] - lateral * self.radius)),
            int(round(self.center[1] - forward * self.radius)),
        )

    def render(
        self,
        current: Sequence[float],
        history: Iterable[Sequence[float]] = (),
    ) -> bytes:
        image = bytearray(self._static)
        points = [self._point(value) for value in history]
        if points:
            count = max(1, len(points) - 1)
            for index, (left, right) in enumerate(zip(points, points[1:])):
                alpha = int(45 + 125 * (index + 1) / count)
                self._line(
                    image,
                    left,
                    right,
                    (80, 235, 180, min(alpha, 180)),
                    width=2,
                )

        endpoint = self._point(current)
        self._line(
            image,
            self.center,
            endpoint,
            (255, 190, 45, 235),
            width=4,
        )
        dx = endpoint[0] - self.center[0]
        dy = endpoint[1] - self.center[1]
        length = math.hypot(dx, dy)
        if length > 8.0:
            unit_x = dx / length
            unit_y = dy / length
            normal_x = -unit_y
            normal_y = unit_x
            base_x = endpoint[0] - 13.0 * unit_x
            base_y = endpoint[1] - 13.0 * unit_y
            left = (
                int(round(base_x + 7.0 * normal_x)),
                int(round(base_y + 7.0 * normal_y)),
            )
            right = (
                int(round(base_x - 7.0 * normal_x)),
                int(round(base_y - 7.0 * normal_y)),
            )
            self._line(image, endpoint, left, (255, 190, 45, 245), width=3)
            self._line(image, endpoint, right, (255, 190, 45, 245), width=3)
        self._disc(image, endpoint[0], endpoint[1], 6, (255, 215, 70, 250))
        return bytes(image)


def iter_stick_hud_frames(
    timeline: FlightTimeline,
    *,
    fps: int,
    video_duration: float,
    lead_seconds: float,
    speed: float,
    trail_seconds: float = 1.5,
    renderer: Optional[StickHudRenderer] = None,
):
    """Yield enough deterministic RGBA frames to cover an encoded video."""

    if fps <= 0:
        raise ValueError("fps must be positive")
    if video_duration <= 0.0:
        raise ValueError("video_duration must be positive")
    hud = renderer or StickHudRenderer()
    trail_frames = max(1, int(round(float(trail_seconds) * fps)))
    total_frames = int(math.ceil(float(video_duration) * fps)) + 2
    for frame_index in range(total_frames):
        video_time = frame_index / float(fps)
        replay_elapsed = replay_elapsed_for_video_time(
            video_time,
            lead_seconds=lead_seconds,
            speed=speed,
            replay_duration=timeline.duration,
        )
        first_history = max(0, frame_index - trail_frames)
        history = []
        for history_index in range(first_history, frame_index + 1):
            history_time = history_index / float(fps)
            history_elapsed = replay_elapsed_for_video_time(
                history_time,
                lead_seconds=lead_seconds,
                speed=speed,
                replay_duration=timeline.duration,
            )
            history.append(timeline.human_input_at(history_elapsed))
        yield hud.render(timeline.human_input_at(replay_elapsed), history)

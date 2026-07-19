"""ROS-free source metadata for clearance guard decisions."""

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class GuardStatusSource:
    """Exact source header parts evaluated by a clearance guard decision."""

    source_valid: bool = False
    seq: int = 0
    stamp_secs: int = 0
    stamp_nsecs: int = 0
    frame_id: str = ""

    @classmethod
    def from_parts(
        cls,
        source_valid,
        seq,
        stamp_secs,
        stamp_nsecs,
        frame_id,
    ):
        return cls(
            source_valid=bool(source_valid),
            seq=int(seq),
            stamp_secs=int(stamp_secs),
            stamp_nsecs=int(stamp_nsecs),
            frame_id=str(frame_id),
        )

    @classmethod
    def from_header(cls, header, source_valid):
        stamp = header.stamp
        return cls.from_parts(
            source_valid=source_valid,
            seq=getattr(header, "seq", 0),
            stamp_secs=getattr(stamp, "secs", 0),
            stamp_nsecs=getattr(stamp, "nsecs", 0),
            frame_id=getattr(header, "frame_id", ""),
        )

    def with_validity(self, source_valid):
        return replace(self, source_valid=bool(source_valid))

    @property
    def has_source(self):
        return bool(
            self.source_valid
            or self.seq
            or self.stamp_secs
            or self.stamp_nsecs
            or self.frame_id
        )

    @property
    def stamp_seconds(self):
        if not self.has_source:
            return float("nan")
        return float(self.stamp_secs) + float(self.stamp_nsecs) * 1.0e-9

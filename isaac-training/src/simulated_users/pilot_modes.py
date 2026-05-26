from enum import IntEnum


class IntentMode(IntEnum):
    CRUISE = 0
    MANEUVER = 1
    STATION_KEEP = 2
    IDLE = 3

    @classmethod
    def count(cls) -> int:
        return len(cls)


class ReactMode(IntEnum):
    NONE = 0
    NO_REACT = 1
    LATE_REACT = 2
    EMERGENCY_STOP = 3
    FREEZE = 4
    EVADE = 5
    OVERCORRECT = 6
    SURGE = 7

    @classmethod
    def count(cls) -> int:
        return len(cls)

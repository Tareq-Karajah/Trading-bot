from datetime import datetime, timezone

class DeadManSwitch:
    """
    Dead-man switch implementation for system liveness checking.
    Timeout: 60 seconds.
    """
    HEARTBEAT_TIMEOUT_SECONDS: int = 60

    def __init__(self) -> None:
        self._last_beat: datetime = datetime.now(timezone.utc)

    def beat(self) -> None:
        """Record current timestamp as last heartbeat."""
        self._last_beat = datetime.now(timezone.utc)

    def check(self, now_utc: datetime) -> bool:
        """
        Returns True if heartbeat is alive.
        Returns False if silent > 60 seconds.
        """
        # Ensure UTC
        if now_utc.tzinfo is None:
            now_utc = now_utc.replace(tzinfo=timezone.utc)
            
        elapsed = (now_utc - self._last_beat).total_seconds()
        return elapsed <= self.HEARTBEAT_TIMEOUT_SECONDS

    def seconds_since_last_beat(self, now_utc: datetime) -> float:
        """Returns seconds elapsed since last heartbeat."""
        if now_utc.tzinfo is None:
            now_utc = now_utc.replace(tzinfo=timezone.utc)
        return (now_utc - self._last_beat).total_seconds()

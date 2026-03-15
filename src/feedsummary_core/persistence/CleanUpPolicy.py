from dataclasses import dataclass


@dataclass(frozen=True)
class CleanupPolicy:
    """Retention windows and scheduling flags for store cleanup jobs."""

    enabled: bool = True
    run_every_minutes: int = 60

    articles_days: int = 180
    daily_summaries_days: int = 7
    weekly_summaries_days: int = 30
    temp_summaries_days: int = 30
    jobs_days: int = 90

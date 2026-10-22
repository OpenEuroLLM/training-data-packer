"""Logging helpers for loguru."""


class DuplicateFilter:
    """Suppress repeated log messages from the same code location.

    Use an instance as a loguru handler filter. Only the first message logged from each file and line is let
    through, even if later messages from that location have different content.
    """

    def __init__(self):
        """Create a filter that has not seen any log locations yet."""
        self.seen = set()

    def __call__(self, record):
        """Decide whether a log record should be emitted.

        :param record: the loguru record to check
        :returns: ``True`` if this is the first record from its file and line, ``False`` otherwise
        """
        # Using the log location/line number prevents repeating messages from the same code line
        log_key = (record["file"].name, record["line"])

        if log_key in self.seen:
            return False  # Suppress log

        self.seen.add(log_key)
        return True  # Allow log

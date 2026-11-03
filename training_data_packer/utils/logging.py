"""Logging helpers for loguru."""

import os

from loguru import logger

GIT_SHA_ENV = "PACKER_GIT_SHA"


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


def log_git_sha() -> None:
    """Log the commit SHA the running code was built from.

    The SHA is read from the ``PACKER_GIT_SHA`` environment variable, which is baked into container images built by
    CI. When the variable is unset, for example when running from a local checkout, a warning that this is not an
    official build is logged instead.
    """
    git_sha = os.environ.get(GIT_SHA_ENV)
    if git_sha:
        logger.info(f"Running training-data-packer commit {git_sha}")
    else:
        logger.warning("Running training-data-packer without a commit SHA, this is not an official build")

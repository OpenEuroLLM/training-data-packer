import os
import unittest
from types import SimpleNamespace
from unittest import mock

from loguru import logger
from parameterized import parameterized

from training_data_packer.utils.logging import GIT_SHA_ENV, DuplicateFilter, log_git_sha


def _record(file_name: str, line: int, message: str = "message") -> dict:
    """Build a minimal loguru-like record.

    :param file_name: name of the file the log call came from
    :param line: line number of the log call
    :param message: the log message
    :returns: a record dict with the fields used by DuplicateFilter
    """
    return {"file": SimpleNamespace(name=file_name), "line": line, "message": message}


class TestDuplicateFilter(unittest.TestCase):
    def test_first_record_is_allowed(self):
        duplicate_filter = DuplicateFilter()

        self.assertTrue(duplicate_filter(_record("a.py", 10)))

    def test_repeated_location_is_suppressed(self):
        duplicate_filter = DuplicateFilter()

        self.assertTrue(duplicate_filter(_record("a.py", 10)))
        self.assertFalse(duplicate_filter(_record("a.py", 10)))
        self.assertFalse(duplicate_filter(_record("a.py", 10)))

    def test_repeated_location_with_different_message_is_suppressed(self):
        duplicate_filter = DuplicateFilter()

        self.assertTrue(duplicate_filter(_record("a.py", 10, "first")))
        self.assertFalse(duplicate_filter(_record("a.py", 10, "second")))

    @parameterized.expand(
        [
            ("different_line", ("a.py", 10), ("a.py", 11)),
            ("different_file", ("a.py", 10), ("b.py", 10)),
            ("different_file_and_line", ("a.py", 10), ("b.py", 20)),
        ],
    )
    def test_different_location_is_allowed(self, _name, first, second):
        duplicate_filter = DuplicateFilter()

        self.assertTrue(duplicate_filter(_record(*first)))
        self.assertTrue(duplicate_filter(_record(*second)))

    def test_instances_do_not_share_state(self):
        first_filter = DuplicateFilter()
        second_filter = DuplicateFilter()

        self.assertTrue(first_filter(_record("a.py", 10)))
        self.assertTrue(second_filter(_record("a.py", 10)))

    def test_seen_tracks_file_name_and_line(self):
        duplicate_filter = DuplicateFilter()

        duplicate_filter(_record("a.py", 10))
        duplicate_filter(_record("b.py", 20))
        duplicate_filter(_record("a.py", 10))

        self.assertEqual({("a.py", 10), ("b.py", 20)}, duplicate_filter.seen)

    def test_filter_with_loguru_suppresses_repeated_log_line(self):
        messages = []
        handler_id = logger.add(messages.append, filter=DuplicateFilter(), format="{message}")
        self.addCleanup(logger.remove, handler_id)

        for i in range(3):
            logger.info(f"repeated {i}")
        logger.info("other line")

        self.assertEqual(["repeated 0\n", "other line\n"], [str(m) for m in messages])


class TestLogGitSha(unittest.TestCase):
    def _capture(self) -> list:
        """Collect log messages for the rest of the test.

        :returns: the list that logged messages are appended to
        """
        messages = []
        handler_id = logger.add(messages.append, format="{message}")
        self.addCleanup(logger.remove, handler_id)
        return messages

    def test_logs_sha_when_set(self):
        messages = self._capture()

        with mock.patch.dict(os.environ, {GIT_SHA_ENV: "abc123"}):
            log_git_sha()

        self.assertEqual(["Running training-data-packer commit abc123\n"], [str(m) for m in messages])

    @parameterized.expand(
        [
            ("unset", {}),
            ("empty", {GIT_SHA_ENV: ""}),
        ],
    )
    def test_warns_unofficial_build_without_sha(self, _name, env):
        messages = self._capture()

        with mock.patch.dict(os.environ, env, clear=True):
            log_git_sha()

        self.assertEqual(
            ["Running training-data-packer without a commit SHA, this is not an official build\n"],
            [str(m) for m in messages],
        )
        self.assertEqual("WARNING", messages[0].record["level"].name)

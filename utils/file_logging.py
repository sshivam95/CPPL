"""Keeping logs."""
import logging
from logging.handlers import BaseRotatingHandler
from logging.handlers import RotatingFileHandler


class MyRotatingFileHandler(RotatingFileHandler):
    """Keep logs of file handling."""

    def __init__(self, filename: str, mode: str = "w", max_bytes: int = 0, backup_count: int = 0) -> None:
        # pylint: disable=non-parent-init-called,super-init-not-called
        """
        Initialize custom file handler.

        :param filename: Name of the file to write the logs in.
        :param mode: default="w", The mode of the logger
        :param max_bytes: default=0, Maximum bytes a file can have
        :param backup_count: default=0, Number of backups generated for the file
        """
        BaseRotatingHandler.__init__(self,
                                     filename,
                                     mode,
                                     encoding=None,
                                     delay=False)
        self.maxBytes = max_bytes
        self.backupCount = backup_count


def tracking_files(filename: str, logger_name: str, level: str) -> logging.getLogger:
    # pylint: disable=attribute-defined-outside-init
    """
    Track all files.

    :param filename: Name of the file used for logging.
    :param logger_name: Logger name.
    :param level: Logger level.
    :return: logger: A logger with the specified logger name.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    filehandler_dbg = MyRotatingFileHandler(
        filename,
        mode="w",
        max_bytes=1,
        backup_count=0
    )  # Create a file logging object of the MyRotatingFileHandler class.
    filehandler_dbg.setLevel(level)
    filehandler_dbg.suffix = ""
    stream_formatter = logging.Formatter(fmt="%(message)s")
    filehandler_dbg.setFormatter(stream_formatter)
    logger.addHandler(filehandler_dbg)
    return logger

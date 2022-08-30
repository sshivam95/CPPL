"""Keeping logs."""
import logging
from logging.handlers import BaseRotatingHandler
from logging.handlers import RotatingFileHandler


class MyRotatingFileHandler(RotatingFileHandler):
    """Keep logs of file handling."""

    def __init__(self, filename, mode="w", maxBytes=0, backupCount=0):
        # pylint: disable=non-parent-init-called,super-init-not-called
        """Initialize custom file handler.

        Parameters
        ----------
        filename
            Name of the file.
        mode
        maxBytes
        backupCount
        """
        BaseRotatingHandler.__init__(self, filename, mode, encoding=None, delay=False)
        self.maxBytes = maxBytes
        self.backupCount = backupCount


def tracking_files(filename, logger_name, level):
    # pylint: disable=attribute-defined-outside-init
    """Track all files.

    Parameters
    ----------
    filename
        Location of the file to log.
    logger_name
        .py file to be logged.
    level
        Logging level.

    Returns
    -------
    logger: object
        Logging object.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    filehandler_dbg = MyRotatingFileHandler(
        filename, mode="w", maxBytes=1, backupCount=0
    )
    filehandler_dbg.setLevel(level)
    filehandler_dbg.suffix = ""
    stream_formatter = logging.Formatter(fmt="%(message)s")
    filehandler_dbg.setFormatter(stream_formatter)
    logger.addHandler(filehandler_dbg)

    return logger

import logging
from logging.handlers import TimedRotatingFileHandler
from logging.handlers import RotatingFileHandler, BaseRotatingHandler

class MyTimedRotatingFileHandler(TimedRotatingFileHandler):
    def doRollover(self):
        x= 1

class MyRotatingFileHandler(RotatingFileHandler):
    def __init__(self, filename, mode='w', maxBytes=0, backupCount=0):
        BaseRotatingHandler.__init__(self, filename, mode, encoding=None, delay=False)
        self.maxBytes = maxBytes
        self.backupCount = backupCount

def tracking_files(filename, logger_name, level):

	logger = logging.getLogger(logger_name) 
	logger.setLevel(level)
	filehandler_dbg = MyRotatingFileHandler(filename, mode='w', maxBytes=1, backupCount=0)
	filehandler_dbg.setLevel(level) 
	filehandler_dbg.suffix = ""
	streamformatter = logging.Formatter(fmt='%(message)s')
	filehandler_dbg.setFormatter(streamformatter)
	logger.addHandler(filehandler_dbg)

	return logger
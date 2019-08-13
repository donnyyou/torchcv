#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Logging tool implemented with the python Package logging.


import argparse
import logging
import os
import sys


DEFAULT_LOG_LEVEL = 'info'
DEFAULT_LOG_FORMAT = '%(asctime)s %(levelname)-7s %(message)s'

LOG_LEVEL_DICT = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR,
    'critical': logging.CRITICAL
}


class Logger(object):
    """
    Args:
      Log level: CRITICAL>ERROR>WARNING>INFO>DEBUG.
      log format: The format of log messages.
    """
    logger = None

    @staticmethod
    def init(log_format=DEFAULT_LOG_FORMAT,
             log_level=DEFAULT_LOG_LEVEL,
             distributed_rank=0):
        assert Logger.logger is None
        Logger.logger = logging.getLogger()
        if distributed_rank > 0:
            return

        if log_level not in LOG_LEVEL_DICT:
            print('Invalid logging level: {}'.format(log_level))
            return

        Logger.logger.setLevel(LOG_LEVEL_DICT[log_level])
        fmt = logging.Formatter(log_format)
        console = logging.StreamHandler()
        console.setLevel(LOG_LEVEL_DICT[log_level])
        console.setFormatter(fmt)
        Logger.logger.addHandler(console)

    @staticmethod
    def check_logger():
        if Logger.logger is None:
            Logger.init(log_level=DEFAULT_LOG_LEVEL, log_format=DEFAULT_LOG_FORMAT)

    @staticmethod
    def debug(message):
        Logger.check_logger()
        filename = os.path.basename(sys._getframe().f_back.f_code.co_filename)
        lineno = sys._getframe().f_back.f_lineno
        prefix = '[{}, {}]'.format(filename,lineno)
        Logger.logger.debug('{} {}'.format(prefix, message))

    @staticmethod
    def info(message):
        Logger.check_logger()
        filename = os.path.basename(sys._getframe().f_back.f_code.co_filename)
        lineno = sys._getframe().f_back.f_lineno
        prefix = '[{}, {}]'.format(filename,lineno)
        Logger.logger.info('{} {}'.format(prefix, message))

    @staticmethod
    def warn(message):
        Logger.check_logger()
        filename = os.path.basename(sys._getframe().f_back.f_code.co_filename)
        lineno = sys._getframe().f_back.f_lineno
        prefix = '[{}, {}]'.format(filename,lineno)
        Logger.logger.warn('{} {}'.format(prefix, message))

    @staticmethod
    def error(message):
        Logger.check_logger()
        filename = os.path.basename(sys._getframe().f_back.f_code.co_filename)
        lineno = sys._getframe().f_back.f_lineno
        prefix = '[{}, {}]'.format(filename,lineno)
        Logger.logger.error('{} {}'.format(prefix, message))

    @staticmethod
    def critical(message):
        Logger.check_logger()
        filename = os.path.basename(sys._getframe().f_back.f_code.co_filename)
        lineno = sys._getframe().f_back.f_lineno
        prefix = '[{}, {}]'.format(filename,lineno)
        Logger.logger.critical('{} {}'.format(prefix, message))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_level', default=None, type=str,
                        dest='log_level', help='To set the level to print to screen.')
    parser.add_argument('--log_format', default="%(asctime)s %(levelname)-7s %(message)s",
                        type=str, dest='log_format', help='The format of log messages.')

    args = parser.parse_args()
    Logger.init(log_level=args.log_level, log_format=args.log_format)

    Logger.info("info test.")
    Logger.debug("debug test.")
    Logger.warn("warn test.")
    Logger.error("error test.")
    Logger.debug("debug test.")

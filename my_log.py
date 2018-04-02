import logging
import sys

_file_logger_name = "results_logger"


def init_file_logging(file_log_level, file_name):
    log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    # file_logger = logging.getLogger(_file_logger_name)
    file_logger = logging.getLogger()
    file_logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(file_name)
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(file_log_level)
    file_logger.addHandler(file_handler)
    _handle_unhandled_exceptions()


def to_file(level, message):
    # logging.getLogger(_file_logger_name).log(level, message)
    logging.getLogger().log(level, message)


def _handle_unhandled_exceptions():
    # Log general unhandled exceptions
    sys.excepthook = _log_unhandled_exception


def _log_unhandled_exception(exc_type, exc_value, exc_traceback):
    """ Handler for catching uncaught exceptions in the whole system """
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logging.getLogger().error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

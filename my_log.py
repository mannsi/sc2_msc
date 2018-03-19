import logging

_file_logger_name = "results_logger"


def init_file_logging(file_log_level, file_name):
    log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_logger = logging.getLogger(_file_logger_name)
    file_logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler("{0}".format(file_name))
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(file_log_level)
    file_logger.addHandler(file_handler)


def to_file(level, message):
    logging.getLogger(_file_logger_name).log(level, message)


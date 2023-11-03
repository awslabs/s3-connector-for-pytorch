import logging

TRACE = 5


def _install_trace_logging():
    logging.addLevelName(TRACE, "TRACE")

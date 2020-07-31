"""logger.py"""
import logging
import settings
import msgpack
import traceback

from datetime import datetime
from pytz import timezone, utc
from fluent import handler
from io import BytesIO

def custom_time(*args):
    utc_dt = utc.localize(datetime.utcnow())
    my_tz = timezone(settings.TIMESTAMP)
    converted = utc_dt.astimezone(my_tz)
    return converted.timetuple()


def set_formatter():
    custom_format = {
        "host": "%(hostname)s",
        "where": "%(module)s.%(funcName)s",
        "type": "%(levelname)s",
        "stack_trace": "%(exc_text)s",
        "timestamp": "%(asctime)s"
    }
    formatter = handler.FluentRecordFormatter(custom_format)
    formatter.converter = custom_time
    return formatter


def overflow_handler(pendings):
    unpacker = msgpack.Unpacker(BytesIO(pendings))
    for unpacked in unpacker:
        print(unpacked)


def get_logger(logger_name):

    print(settings)

    try:

        if isinstance(logger_name, str):
            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger(logger_name)
            logger.level = logging.INFO

            fluent_handler = handler.FluentHandler(
                __name__,
                host=settings.FLUENTD_HOST,
                port=settings.FLUENTD_PORT,
                buffer_overflow_handler=overflow_handler
            )

            formatter = set_formatter()
            fluent_handler.setFormatter(formatter)

            logger.addHandler(fluent_handler)
            return logger

        raise Exception("logger_name must be a string")
    
    except Exception as exp:
        traceback.print_stack()
        logging.error(str(exp))        
        return None
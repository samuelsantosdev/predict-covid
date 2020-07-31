import os
import logging
import sys

logger = logging.basicConfig(level=logging.INFO, format='%(message)s')

try:
    
    #GLOBAL CONFIG
    TIMESTAMP       = os.environ['TIMESTAMP']
    
    #Variables to train model
    URL_DATASOURCE  = os.environ['URL_DATASOURCE']
    DATALAKE        = os.environ['DATALAKE']
    ALL_COLS        = os.environ['ALL_COLS'].split(',')
    DAYS_MEAN       = int(os.environ['DAYS_MEAN'])
    START_DATE_DATA = os.environ['START_DATE_DATA']
    ESTIMATORS      = int(os.environ['ESTIMATORS'])
    N_JOBS          = int(os.environ['N_JOBS'])
    RANDOM_STATE    = int(os.environ['RANDOM_STATE'])
    PREDICT_DAYS    = int(os.environ['PREDICT_DAYS'])
    TEST_VARIABLES  = bool(int(os.environ['TEST_VARIABLES']))
    COLUMNS_PREDICT = list(os.environ['COLUMNS_PREDICT'].split(','))

    #Logs to elastic
    FLUENTD_HOST    = os.environ['FLUENTD_HOST']
    FLUENTD_PORT    = int(os.environ['FLUENTD_PORT'])
    
except Exception as ex:
    logger.error("Environment variables is missing - {}".format(ex))
    sys.exit(1)

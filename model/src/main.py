''' Main module to train'''
import settings
import pandas as pd
import traceback
import os
import pickle
import shutil
import schedule
import time

from utils.logger import get_logger
from train.predict import Predict
from utils.datasource import Datasource
from datetime import datetime, timedelta

logger = get_logger('predict_covid')

def main():
    '''Receive functions'''

    try:
        
        today = datetime.now().strftime('%Y-%m-%d')
        file_datasource = '{}/sources/datasource-{}.csv'.format(settings.DATALAKE, today)

        datasource      = Datasource()

        logger.info("Downloading data csv")
        if os.path.isfile(file_datasource) == False :
            datasource_file = datasource.from_url(url=settings.URL_DATASOURCE, file_path=file_datasource) 
        else:
            datasource_file = file_datasource

        logger.info("Convert to Dataframe")
        df              = pd.read_csv(datasource_file)

        logger.info("Training model")
        predict = Predict()
        model = predict.train_model(df)
        if model :
            predict_file = '{}/history/models/{}_predict.pkl'.format(settings.DATALAKE, today)
            with open( predict_file, 'wb') as file:
                pickle.dump(model, file)

            shutil.copy(predict_file, '{}/results/predict.pkl'.format(settings.DATALAKE))
        
        data = predict.predict(settings.PREDICT_DAYS)
        if data :
            df_predicted = pd.DataFrame(data)

            parquet_file = '{}/history/parquets/{}_forecast_{}_days.parquet.gzip'.format(settings.DATALAKE, today, settings.PREDICT_DAYS)
            df_predicted.to_parquet(parquet_file, compression='gzip')
            
            shutil.copy(parquet_file, '{}/results/forecast.parquet'.format(settings.DATALAKE))
    
    except Exception as exp:
        traceback.print_tb(exp.__traceback__)
        logger.error('Error on train model %s' % str(exp) )        

if __name__ == "__main__":
    main()
    # TODO: run un production
    # schedule.every().day.at("04:00").do(main)
    # while True:
    #     schedule.run_pending()
    #     time.sleep(1)

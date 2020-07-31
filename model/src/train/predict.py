'''Predict'''
import pandas as pd
import numpy as np
import seaborn as sns
import requests
import traceback
import settings
import random
import itertools
import json
import pickle
import os

from utils.logger import get_logger
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_log_error
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta

logger = get_logger('predict_covid')

class Predict():
    """Predict class """

    __best_columns  = None
    __best_pred     = 0
    __used_columns  = []

    __model = None
    __predict_evolution = 0

    __precit_dates  = []
    __precit_dates  = []
    __valid_dataframe = None

    def __init__(self):
        self.__best_columns = self.__get_best_columns()
        self.__best_pred    = 1000
        self.__model        = self.__get_regressor()

    def __get_best_columns(self):
        """Load the most best available columns to predict"""
        try:
            #Load columns saved in datalake
            with open( settings.DATALAKE+'/results/columns.json', r ) as f:
                best_columns = json.loads(f.read())
            logger.info('Columns by json')
            return best_columns
        except Exception as exp:
            logger.info('Columns by env')
            return settings.COLUMNS_PREDICT


    def __save_plot_image(self, plot, filename: str)->None:
        """Save plots with png"""
        fig = plot.get_figure()
        plot.grid(True)
        fig.savefig("{}{}".format(settings.DATALAKE, filename))

    def __get_regressor(self)->RandomForestRegressor:
        return RandomForestRegressor(n_jobs=settings.N_JOBS, random_state=settings.RANDOM_STATE, n_estimators=settings.ESTIMATORS)

    def __validate_variables(self, Xtr: pd.DataFrame, Xval: pd.DataFrame, yval: pd.DataFrame, ytr:pd.DataFrame, vars:list)->None:
        """Score a list of variables to prediciton """

        self.__model.fit(Xtr[vars], ytr)
        p = self.__model.predict(Xval[vars])
        
        p_final = Xval['current_new_cases'] + p
        yval_final = Xval['current_new_cases'] + yval

        erro = np.sqrt(mean_squared_log_error(yval_final, p_final))
        
        if erro < self.__best_pred :
            self.__best_pred = erro
            self.__best_columns = vars
        
        logger.info("erro={:.4f} - var={}".format(erro, ",".join(vars)))
        

    def dataframe_to_train(self, covid_train: pd.DataFrame, covid_valid: pd.DataFrame):
        """Prepare data to train"""

        df_X_treino = pd.DataFrame()
        df_X_valid = pd.DataFrame()

        df_X_treino['diff_new_cases_next_day'] = covid_train['new_cases'].diff().shift(-1)
        df_X_valid['diff_new_cases_next_day'] = covid_valid['new_cases'].diff().shift(-1)

        # Variables Seasonal
        logger.info('Seasonal')
        df_X_treino['month'] = covid_train['date'].dt.month
        df_X_treino['day'] = covid_train['date'].dt.day
        df_X_treino['weekday'] = covid_train['date'].dt.weekday

        df_X_valid['month'] = covid_valid['date'].dt.month
        df_X_valid['day'] = covid_valid['date'].dt.day
        df_X_valid['weekday'] = covid_valid['date'].dt.weekday

        # Variables Lag
        logger.info('Lag')
        df_X_treino['current_new_cases'] = covid_train['new_cases']
        df_X_valid['current_new_cases'] = covid_valid['new_cases']

        # Variables Diff lag
        logger.info('Diff lag')
        df_X_treino['diff_current_new_cases'] = covid_train['new_cases'].diff()
        df_X_valid['diff_current_new_cases'] = covid_valid['new_cases'].diff()

        #Variables mean
        logger.info('Var MEAN')
        days = settings.DAYS_MEAN
        df_X_valid['mean_current_days'] = covid_valid['new_cases'].rolling(days).mean()
        df_X_treino['mean_current_days'] = covid_train['new_cases'].rolling(days).mean()

        #Remove lines with na values
        df_X_treino = df_X_treino.dropna()
        df_X_valid = df_X_valid.dropna()

        logger.info('Columns to predict')
        logger.info(",".join(df_X_treino.columns))
        
        #Dataframe to train and validate
        Xtr, ytr = df_X_treino.drop(['diff_new_cases_next_day'], axis=1), df_X_treino['diff_new_cases_next_day']
        Xval, yval = df_X_valid.drop(['diff_new_cases_next_day'], axis=1), df_X_valid['diff_new_cases_next_day']

        return Xtr, ytr, Xval, yval



    def testing_variables(self, Xtr: pd.DataFrame, ytr: pd.DataFrame, Xval: pd.DataFrame, yval: pd.DataFrame):
        """Test all variables to know the best sequence to predict """
        logger.info('Testing all combinations of variables')
        
        #all columns
        columns = Xtr.columns
        for num, col in enumerate(columns):

            #all possible combinations
            combinations = list(itertools.combinations(columns, num+1))
            for comb in combinations:

                #all possible lengths of combinations
                permutations = list(itertools.permutations( list(comb) , len(list(comb))))
                for key, permutate in enumerate(permutations): 
                    
                    #columns to predict
                    cols = list(permutate)
                    if cols not in self.__used_columns:
                        self.__validate_variables(Xtr, Xval, yval, ytr, cols)
                        self.__used_columns.append(cols)
        
        #save best combination of columns
        with open( settings.DATALAKE+'/results/columns.json', 'wb' ) as f:
            f.write( bytes(json.dumps(self.__best_columns), encoding='utf-8') )
            
    def __prepare_data(self, data_frame: pd.DataFrame):
        """Prepare all data, colmns and types """
        data_frame['date'] = pd.to_datetime(data_frame['date'])
                
        le = LabelEncoder()
        
        le.fit(data_frame['iso_code'].astype(str))
        data_frame['iso_code'] = le.transform(data_frame['iso_code'].astype(str))

        le.fit(data_frame['continent'].astype(str))
        data_frame['continent'] = le.transform(data_frame['continent'].astype(str))

        df_no_nan = data_frame.apply(lambda x: x.replace([np.NaN], '0', regex=True) )
        df_no_nan = df_no_nan[settings.ALL_COLS].replace('[\D]', '0', regex=True)

        df_datetime = df_no_nan.copy()
        df_datetime['new_cases'] = df_datetime['new_cases'].replace('[\D]', '0', regex=True)
        df_datetime['new_cases'] = df_datetime['new_cases'].astype(int)
        df_datetime['aged_65_older'] = df_datetime['aged_65_older'].astype(int)
        df_datetime['aged_70_older'] = df_datetime['aged_70_older'].astype(int)
        df_datetime = df_datetime.sort_values(['date'], ascending=True).reset_index()

        plot = df_datetime.plot(x='date', y='new_cases', title = "Original data", figsize=(6,3))
        self.__save_plot_image(plot, "/images/original_data.png")
        
        df_datetime = df_datetime.groupby(['date'])['new_cases'].agg('sum').reset_index()

        plot = df_datetime.plot(x='date', y='new_cases', title = "Data grouped by date", figsize=(12,4))
        self.__save_plot_image(plot, "/images/grouped_data.png")

        datetime_end = datetime.now() - timedelta(days=1)
        datetime_start_valid = datetime.now() - timedelta(days=22)
        
        covid_train = df_datetime[ 
            (df_datetime['date'] > settings.START_DATE_DATA) & 
            (df_datetime['date'] < datetime_start_valid.strftime('%Y-%m-%d')) & 
            (df_datetime['new_cases'] >= 0)
        ]

        plot = covid_train.plot.line(x='date', y='new_cases', title="Train data {} {}".format(settings.START_DATE_DATA, datetime_start_valid.strftime('%Y-%m-%d')) ,figsize=(6,3))
        self.__save_plot_image(plot, "/images/train_data.png")
        
        covid_valid = df_datetime[ 
            (df_datetime['date'] >= datetime_start_valid.strftime('%Y-%m-%d') ) & 
            (df_datetime['date'] <= datetime_end.strftime('%Y-%m-%d') ) & 
            (df_datetime['new_cases'] >= 0)
        ]

        plot = covid_valid.plot.line(x='date', y='new_cases', title="Valid data {} {}".format(datetime_start_valid.strftime('%Y-%m-%d'), datetime_end.strftime('%Y-%m-%d')) ,figsize=(6,3))
        self.__save_plot_image(plot, "/images/valid_data.png")

        covid_train = covid_train.dropna()
        covid_valid = covid_valid.dropna()

        #baseline
        y_treino = covid_train['new_cases']
        y_valid = covid_valid['new_cases']
        baseline_train = covid_train['new_cases'].shift(1)
        baseline_valid = covid_valid['new_cases'].shift(1)

        covid_train['baseline'] = baseline_train
        covid_valid['baseline'] = baseline_valid

        erro_baseline = np.sqrt(mean_squared_log_error(y_valid[baseline_valid.notnull()], baseline_valid[ baseline_valid.notnull() ] ))
        logger.info('Mean_squared_log_error baseline {}'.format(erro_baseline*100))

        covid_train['diff_new_cases'] = covid_train['new_cases'].diff().shift(-1)
        covid_valid['diff_new_cases'] = covid_valid['new_cases'].diff().shift(-1)

        plot = covid_train.plot(x='date', y='diff_new_cases', title = "Data oscilation", figsize=(6,3))
        self.__save_plot_image(plot, "/images/normalized_diff_data.png")

        return covid_train, covid_valid, erro_baseline

    def predict(self, days: int)->pd.DataFrame:
        """Predict by days"""

        dates_to_predict    = []
        dates_forecast      = []
        datetimes           = []

        if isinstance(days,int) and days > 0 and \
            os.path.isfile(settings.DATALAKE+'/results/base_to_calc.parquet') == False :
            raise Exception("You need train the model, run $ python3 src/main.py")

        self.__valid_dataframe = pd.read_parquet(settings.DATALAKE+'/results/base_to_calc.parquet', engine='fastparquet')
        calc_cases          = int(self.__valid_dataframe['current_new_cases'].iloc[[-1]].values[0])
        
        logger.info('Prepare days to predict')
        for day in range(0, days ):
            _datetime = datetime.now() + timedelta(days=day)
            datetimes.append(_datetime)
            dates_to_predict.append([_datetime.month, _datetime.weekday(), _datetime.day])

        logger.info('Predict all days')
        self.__model = pickle.load(open(settings.DATALAKE+'/results/predict.pkl', 'rb'))
            
        for key, date in enumerate(dates_to_predict):
            calc_cases += self.__model.predict([date])[0]
            print("{} -> {:.2f}".format(key+1, calc_cases))
            dates_forecast.append({ "date" : datetimes[key].strftime('%Y-%m-%d'), "new_cases" : calc_cases})

        covid_forecast = pd.DataFrame(dates_forecast)
        plot = covid_forecast.plot(x='date', y='new_cases', title = "Forecast", figsize=(12,4))
        self.__save_plot_image(plot, "/images/forecast_data.png")

        return dates_forecast

    def train_model(self, data_frame: pd.DataFrame)->RandomForestRegressor:
        """Train a model with RandomForestRegressor """
        try:
            
            logger.info('Dataframe prepare')
            covid_train, convid_valid, erro_baseline = self.__prepare_data(data_frame)

            logger.info('Create dataframe to train')
            Xtr, ytr, Xval, yval = self.dataframe_to_train(covid_train, convid_valid)
            
            self.__valid_dataframe = Xval
            self.__valid_dataframe.to_parquet(settings.DATALAKE+'/results/base_to_calc.parquet', compression='gzip')

            logger.info('Test variables {}'.format(settings.TEST_VARIABLES))
            if settings.TEST_VARIABLES :
                self.testing_variables(Xtr, ytr, Xval, yval)
                logger.info('Result test variables {} - {}'.format( str(self.__best_pred) , ",".join(self.__best_columns) ) )
            else:
                self.__validate_variables(Xtr, Xval, yval, ytr, self.__best_columns)

            logger.info('Start train RandomForestRegressor')
            
            self.__model.fit(Xtr[ self.__best_columns ], ytr)

            logger.info('Mean_squared_log_error baseline {}'.format(erro_baseline*100))
            logger.info('Mean_squared_log_error predict {}'.format(self.__best_pred*100))

            return self.__model

        except Exception as exp:
            traceback.print_tb(exp.__traceback__)
            logger.error(str(exp))
            return None



    
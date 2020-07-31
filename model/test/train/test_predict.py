import pytest
from pandas import DataFrame
from train.predict import Predict

class TestTrain():

    def test_train_error(self):
        assert Predict().train_model(None) == None
        assert Predict().train_model(None) == None
        assert Predict().train_model(0) == None
        assert Predict().train_model(0.1) == None

    def test_train_success(self):

        df = DataFrame()
        assert Predict().train_model(df) == None
        

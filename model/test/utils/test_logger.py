import pytest
import logging

from utils.logger import get_logger

class TestLogger():

    def test_logger_error(self):
        assert get_logger(None) == None
        assert get_logger(1) == None
        assert get_logger(1.3) == None
        assert get_logger(False) == None

    def test_logger_success(self):
        
        assert get_logger('test_logger') != None
        assert get_logger('test_logger') != ''
        

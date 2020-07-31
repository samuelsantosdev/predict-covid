'''Datasource provider '''
import requests
import traceback

from utils.logger import get_logger
logger = get_logger('predict_covid')

class Datasource():
    """ Provides all datasources """

    def from_url(self, url: str, file_path: str):
        ''' Request a file to use as a datasource'''
        try:   

            resp = requests.get(url)
            with open( file_path, 'wb') as file:
                file.write(resp.content)

            return file_path

        except Exception as exc:
            traceback.print_stack()
            logger.error('Error on train model {}'.format(str(exc)) )
            return None

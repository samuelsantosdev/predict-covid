import pytest
import mock
from utils.datasource import Datasource

class TestDatasource():

    __datasource    = None
    mock_response = None

    def setup_class(self):
        self.__datasource = Datasource()
        self.mock_response = mock.Mock()
        self.mock_response.content = bytes('teste', encoding='utf8')

    @mock.patch('requests.get', mock.Mock(
        side_effect=lambda k:
            {'url': TestDatasource().mock_response }.get(k, 'unhandled request %s'%k)
        ))
    @pytest.mark.parametrize("url, file_name, expected", [
        ('url', '/tmp/teste.csv', '/tmp/teste.csv'),
        ('url', None, None),
        ('url', '', None),
        ('url2', None, None),
        ('url3', '', None),
    ])
    def test_from_url(self, url, file_name, expected):

        file = self.__datasource.from_url(url, file_name)

        assert file == expected
        
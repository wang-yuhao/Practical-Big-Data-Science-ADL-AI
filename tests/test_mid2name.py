import pytest
import pickle
from os import pardir
from os.path import join


@pytest.fixture
def test_mid_dictionaries(self) -> None:
    """ checks whether the dictionary mappings (entities) are correct
    """
    path = join(pardir, 'data/processed/FB15k-237/', 'mid2name.pkl')
    loaded_data = pickle.load(open(path, "rb"))
    assert loaded_data['/m/06kxk2'] == "Carl Foreman"
    assert loaded_data['/m/02xry'] == "Florida"
    assert loaded_data['/m/02hcv8'] == "Eastern Time Zone"
    assert loaded_data['/m/01trf3'] == "Martin Short"
    assert loaded_data['/m/0np9r'] == "voice actor"

import pytest
from fastapi.testclient import TestClient
from app.data import generate
from app.ml import train
from app.main import create_app

@pytest.fixture(scope='session')
def experiment():return train(generate())

@pytest.fixture(scope='session')
def client(experiment):
    with TestClient(create_app(experiment),base_url='http://127.0.0.1:8010') as c:yield c

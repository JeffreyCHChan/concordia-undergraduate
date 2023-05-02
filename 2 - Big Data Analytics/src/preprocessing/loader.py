import os
import shutil

import pandas as pd
import requests
from requests import HTTPError
from requests.auth import HTTPBasicAuth

from src.core import init_spark
from .functions import *

USER = "537"
KEY = "4d37a9f6451950c34118d7b983438da8"


def _local_load():
    path = os.getcwd() + '/artifacts'
    file = path + '/dataset.pkl'

    if not os.path.isfile(file):
        shutil.rmtree(path)
        raise FileNotFoundError('dataset.pkl not found.')

    df = pd.read_pickle(file)
    return df


def _remote_load():
    path = os.getcwd() + '/artifacts'
    file = path + '/dataset.pkl'
    spark = init_spark()
    auth = HTTPBasicAuth(USER, KEY)

    data1 = requests.get(
        'https://opendata.concordia.ca/API/v1/course/catalog/filter/*/*/*',
        auth=auth,

    )
    data2 = requests.get(
        'https://opendata.concordia.ca/API/v1/course/description/filter/*',
        auth=auth
    )

    if data1.status_code != 200 or data2.status_code != 200:
        raise HTTPError('Cannot connect to opendata.concordia.ca')

    df = pd.merge(pd.read_json(data1.text), pd.read_json(data2.text), on=["ID"])
    df = spark.createDataFrame(df)

    df = df.withColumn('description', generate_description(df['description'])) \
        .withColumn('credit', generate_credit(df['classUnit'])) \
        .withColumn('title', generate_title(df['title'])) \
        .withColumn('career', generate_career(df['career'])) \
        .withColumn('code', generate_code(df['subject'], df['catalog'])) \
        .withColumn('requisites', generate_requisites(df['prerequisites'], df['crosslisted']))

    # df = df.withColumn('concepts', get_entities(df['description']))
    df = df.drop('classUnit', 'prerequisites', 'crosslisted')
    df = df.select('ID', 'code', 'title', 'subject', 'catalog', 'career', 'credit', 'requisites', 'description')
    df = df.toPandas()
    df = generate_pandas_concepts(df)
    os.mkdir(path)
    df.to_pickle(file)
    return df


def load() -> pd.DataFrame:
    try:
        df = _local_load()
    except FileNotFoundError:
        df = _remote_load()

    return df

import findspark
from pyspark.sql import SparkSession, DataFrame

findspark.init()
__SPARK = SparkSession.builder.getOrCreate()


def init_spark():
    global __SPARK
    return __SPARK


def to_csv(dataframe: DataFrame, path):
    dataframe.coalesce(1).write.csv(path)
    dataframe.repartition(1).write.csv(path)


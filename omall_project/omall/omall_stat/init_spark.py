from pyspark.sql import SparkSession


def init_spark(app_name):
    spark = SparkSession \
        .builder \
        .appName(app_name) \
        .enableHiveSupport() \
        .getOrCreate()
    sc = spark.sparkContext
    return spark, sc

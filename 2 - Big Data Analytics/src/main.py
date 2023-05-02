from src.core import init_spark
from src.preprocessing.loader import load
import re
from pyspark.sql.functions import udf, regexp_replace, split, col, desc
from pyspark.sql.types import StringType, DoubleType, FloatType
from pyspark.sql.functions import concat_ws
import pickle

# PySPark's Word2Vec for a fixed size vectors
from pyspark.ml.feature import Word2Vec, VectorAssembler, Normalizer, SQLTransformer, BucketedRandomProjectionLSH
from pyspark.ml.feature import CountVectorizer

#Basic Model
from pyspark.ml.clustering import GaussianMixture

pattern = re.compile(r'http:\/\/dbpedia\.org\/resource\/')


def replace_pattern(text):
    return [pattern.sub(r' ', t) for t in text]


if __name__ == '__main__':
    df = load()
    spark = init_spark()
    df = spark.createDataFrame(df)
    # df.printSchema()
    # print(df.count())
    udf_replace_pattern = udf(replace_pattern, StringType())

    df = df.withColumn('concepts_isolated', udf_replace_pattern('concepts'))
    df = df.withColumn("concepts_isolated_no_brackets", regexp_replace("concepts_isolated", r"\[|\]", ""))

    df = df.withColumn('concatenated_columns',
                       concat_ws(" ", "subject", "catalog", "career", "credit", "concepts_isolated_no_brackets"))
    df = df.withColumn("array_features", split("concatenated_columns", ",")  )# puts the concepts into an array for Word2Vec
    # df.select("array_features").show(10, truncate=False)

    # https://spark.apache.org/docs/latest/ml-features.html#countvectorizer
    # https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.CountVectorizer.html
    cv = CountVectorizer(inputCol="array_features", outputCol="featuresCountVec")
    cv.binary = True
    cv.setMinDF = 2
    cv.setMinTF = 1
    model = cv.fit(df)
    result = model.transform(df)
    # result.select("featuresCountVec").show(10, truncate=False)

    # dropping everything except the course code and the countvectorizer
    result = result.drop('ID', 'title', 'subject', 'catalog', 'career', 'credit', 'requisites', 'description', 'concepts', 'concepts_isolated', 'concepts_isolated_no_brackets', 'concatenated_columns', 'array_features')
    # result.show(10, truncate=False)

    # useful to see the overall columns and metadata
    # result.printSchema()

    # training and test splits
    # training, test = df.randomSplit([0.8, 0.2])

    # uncomment for a visualization
    # for row in result.collect():
    #     # className, vector = row[1], row[14]
    #     className, vector = row
    #     print("Course Code: %s => \nVector: %s\n" % (className, str(vector)))



    # Define the cosine similarity function
    def cosine_similarity(vec1, vec2):
        return float(vec1.dot(vec2) / (vec1.norm(2) * vec2.norm(2)))


    # Register the cosine similarity function as a UDF
    cosine_similarity_udf = udf(cosine_similarity, DoubleType())

    # CrossJoin the DataFrame with itself to get all pairs of rows
    df_pairs = result.alias("a").crossJoin(result.alias("b"))

    # Register the UDF
    spark.udf.register("cosine_similarity_udf", cosine_similarity, DoubleType())

    # Calculate the cosine similarity for each pair of rows
    df_similarity = df_pairs.selectExpr(
        "a.code as code1",
        "b.code as code2",
        "cosine_similarity_udf(a.featuresCountVec, b.featuresCountVec) as similarity"
    )

    # It won't display the CosineSimilarity for identical course codes
    # And won't display redundant rows (ex: COMP352 | COMP346 | 0.4
    #                                       COMP346 | COMP352 | 0.4 )
    df_similarity = df_similarity.filter(col("code1") < col("code2"))
    #df_similarity.printSchema()
    # df_similarity.show(50)



    # # I keep gettings error when I try to order the "similarity" column in ASC or DESC fashion
    df_similarity = df_similarity.filter(col("similarity") > 0)
    df_similarity = df_similarity.orderBy('similarity', ascending=False)
    # df_similarity.show(50)
    #
    try:
        df_similarity.toPandas().to_pickle('similarity.pkl')
    except Exception as e:
        print(e)
    with open('similarity.pkl', 'rb') as file:
        df = pickle.load(file)
        df = spark.createDataFrame(df)
        df.show()
    # TODO: DONE create the dataset that concatenates into an array(subject, catalog, career, credit, concepts_isolated) and apply Wrod2Vec.
    # TODO: DONE use a similarity calculation
    # TODO: DONE build the content based recommender using cosine similarity (maybe something else)
    # TODO: use a simple clustering model either in Spark or Sci-kit learn

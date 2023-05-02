import random
import re

from pyspark import Row
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession
from pyspark.sql.functions import col


def init_spark():
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    return spark


def generateSyntheticData():
        spark = init_spark()
        rddCourses = spark.read.text("courses.txt").rdd
        rddAllCourses = spark.read.text("allCourses.txt").rdd
        splitRdd = rddAllCourses.map(lambda x: x.value.split("::"))

        studentsPerProgram = 1

        with open("syntheticData.txt", "a") as f:
            times = 2660
            for courseType in rddCourses.collect():
                print(courseType)
                for student in range(0,studentsPerProgram):
                    for uniqueCourse in splitRdd.collect():
                        if re.search(courseType[0], uniqueCourse[0]):
                            rand = random.randint(1, 10)
                            if rand <= 6:
                                rating = random.uniform(3.0,5.0)
                                f.write(str(student + times) + "::" + str(uniqueCourse[1]) + "::" + str(uniqueCourse[0]) + "::" + str(rating) + "\n")
                        else:
                            rand = random.randint(1,500)
                            if rand == 1:
                                notInProgramRating = random.uniform(1.0, 5.0)
                                f.write(str(student + times) + "::" + str(uniqueCourse[1]) + "::" + str(uniqueCourse[0]) + "::" + str(notInProgramRating) + "\n")
                times = times + studentsPerProgram

def generateCollaborativeFilteringModel():
    spark = init_spark()
    lines = spark.read.text("syntheticData.txt").rdd
    lineArr = lines.map(lambda line: line.value.split("::"))

    ratingsRdd = lineArr.map(
        lambda line: Row(studentId=int(line[0]), classId=int(line[1]), className=str(line[2]), rating=float(line[3])))

    ratingsDataFrame = spark.createDataFrame(ratingsRdd)
    (training, test) = ratingsDataFrame.randomSplit([0.8, 0.2])

    als = ALS(userCol="studentId", itemCol="classId", ratingCol="rating", coldStartStrategy="drop", rank=70, regParam=0.1)
    model = als.fit(training)
    predictions = model.transform(test)

    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")

    rmse = evaluator.evaluate(predictions)

    print("RMSE: " + str(rmse))

    #Example for 1 user.
    #userId = 474
    #userRecs = model.recommendForUserSubset(ratingsDataFrame.filter(col("studentId") == userId), 10)

    users = ratingsDataFrame.select(als.getUserCol()).distinct().limit(3)
    userRecs = model.recommendForUserSubset(users, 10)

#Uncomment to view in text file
    # s = ""
    # with open("classRecommendationsCF.txt", "w") as f:
    #     for rec in userRecs.collect():
    #         user_id = rec[0]
    #         s = user_id
    #         for item in rec["recommendations"]:
    #             item_name = ratingsDataFrame.filter(col("classId") == item[0]).collect()[0][2]
    #             f.write(str(s) + " | " + str(item_name) + " | " + str(item[0]) + " | " + str(item[1]) + "\n")
    #         f.write("\n")

generateSyntheticData()

import warnings
import findspark
findspark.init("C:\spark")
import pyspark
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report
from pyspark.ml.classification import GBTClassifier, LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder, StandardScaler
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql.functions import when, count, col, explode, array, lit
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)


spark = SparkSession.builder \
    .master("local") \
    .appName("pyspark_giris") \
    .getOrCreate()

sc = spark.sparkContext
# sc.stop()

spark_df = spark.read.csv("datasets/churn2.csv", header=True, inferSchema=True)
spark_df
type(spark_df)
# pyspark.sql.dataframe.DataFrame

spark_df.printSchema()
# root
#  |-- RowNumber: integer (nullable = true)
#  |-- CustomerId: integer (nullable = true)
#  |-- Surname: string (nullable = true)
#  |-- CreditScore: integer (nullable = true)
#  |-- Geography: string (nullable = true)
#  |-- Gender: string (nullable = true)
#  |-- Age: integer (nullable = true)
#  |-- Tenure: integer (nullable = true)
#  |-- Balance: double (nullable = true)
#  |-- NumOfProducts: integer (nullable = true)
#  |-- HasCrCard: integer (nullable = true)
#  |-- IsActiveMember: integer (nullable = true)
#  |-- EstimatedSalary: double (nullable = true)
#  |-- Exited: integer (nullable = true)

spark_df.dtypes
spark_df.head(5)
spark_df.show(5)

# Değişken isimlerinin küçültülmesi
spark_df = spark_df.toDF(*[c.lower() for c in spark_df.columns])
spark_df.show(5)

# +---------+----------+--------+-----------+---------+------+---+------+---------+-------------+---------+--------------+---------------+------+
# |rownumber|customerid| surname|creditscore|geography|gender|age|tenure|  balance|numofproducts|hascrcard|isactivemember|estimatedsalary|exited|
# +---------+----------+--------+-----------+---------+------+---+------+---------+-------------+---------+--------------+---------------+------+
# |        1|  15634602|Hargrave|        619|   France|Female| 42|     2|      0.0|            1|        1|             1|      101348.88|     1|
# |        2|  15647311|    Hill|        608|    Spain|Female| 41|     1| 83807.86|            1|        0|             1|      112542.58|     0|
# |        3|  15619304|    Onio|        502|   France|Female| 42|     8| 159660.8|            3|        1|             0|      113931.57|     1|
# |        4|  15701354|    Boni|        699|   France|Female| 39|     1|      0.0|            2|        0|             0|       93826.63|     0|
# |        5|  15737888|Mitchell|        850|    Spain|Female| 43|     2|125510.82|            1|        1|             1|        79084.1|     0|
# +---------+----------+--------+-----------+---------+------+---+------+---------+-------------+---------+--------------+---------------+------+
# only showing top 5 rows


# özet istatistikler
spark_df.describe().show()

# +-------+------------------+-----------------+-------+-----------------+---------+------+------------------+------------------+-----------------+------------------+-------------------+-------------------+-----------------+-------------------+
# |summary|         RowNumber|       CustomerId|Surname|      CreditScore|Geography|Gender|               Age|            Tenure|          Balance|     NumOfProducts|          HasCrCard|     IsActiveMember|  EstimatedSalary|             Exited|
# +-------+------------------+-----------------+-------+-----------------+---------+------+------------------+------------------+-----------------+------------------+-------------------+-------------------+-----------------+-------------------+
# |  count|             10000|            10000|  10000|            10000|    10000| 10000|             10000|             10000|            10000|             10000|              10000|              10000|            10000|              10000|
# |   mean|            5000.5|  1.56909405694E7|   null|         650.5288|     null|  null|           38.9218|            5.0128|76485.88928799961|            1.5302|             0.7055|             0.5151|100090.2398809998|             0.2037|
# | stddev|2886.8956799071675|71936.18612274907|   null|96.65329873613035|     null|  null|10.487806451704587|2.8921743770496837|62397.40520238599|0.5816543579989917|0.45584046447513327|0.49979692845891815|57510.49281769821|0.40276858399486065|
# |    min|                 1|         15565701|  Abazu|              350|   France|Female|                18|                 0|              0.0|                 1|                  0|                  0|            11.58|                  0|
# |    max|             10000|         15815690| Zuyeva|              850|    Spain|  Male|                92|                10|        250898.09|                 4|                  1|                  1|        199992.48|                  1|
# +-------+------------------+-----------------+-------+-----------------+---------+------+------------------+------------------+-----------------+------------------+-------------------+-------------------+-----------------+-------------------+

# sadece belirli değişkenler için özet istatistikler
spark_df.describe(["age", "exited"]).show()


# Kategorik değişken sınıf istatistikleri
spark_df.groupby("exited").count().show()

#  +------+-----+
# |exited|count|
# +------+-----+
# |     1| 2037|
# |     0| 7963|
# +------+-----+

# select(): Değişken seçimi
spark_df.select("age", "surname").show(5)

spark_df[spark_df.age<20].show(4)


# groupby işlemleri
spark_df.groupby("exited").count().show()
# +------+-----+
# |exited|count|
# +------+-----+
# |     1| 2037|
# |     0| 7963|
# +------+-----+

spark_df.groupby("exited").agg({"age": "mean"}).show()
#  +------+-----------------+
# |exited|         avg(age)|
# +------+-----------------+
# |     1| 44.8379970544919|
# |     0|37.40838879819164|

from pyspark.sql.functions import when, count, col



[col for col in spark_df.dtypes]
# [('rownumber', 'int'),
#  ('customerid', 'int'),
#  ('surname', 'string'),
#  ('creditscore', 'int'),
#  ('geography', 'string'),
#  ('gender', 'string'),
#  ('age', 'int'),
#  ('tenure', 'int'),
#  ('balance', 'double'),
#  ('numofproducts', 'int'),
#  ('hascrcard', 'int'),
#  ('isactivemember', 'int'),
#  ('estimatedsalary', 'double'),
#  ('exited', 'int')]


# Tüm numerik değişkenlerin seçimi ve özet istatistikleri
num_cols = [col[0] for col in spark_df.dtypes if col[1] != 'string']


spark_df.select(num_cols).describe().toPandas().transpose()

# summary                max
# rownumber            10000
# customerid        15815690
# creditscore            850
# age                     92
# tenure                  10
# balance          250898.09
# numofproducts            4
# hascrcard                1
# isactivemember           1
# estimatedsalary  199992.48
# exited                   1



# Tüm kategorik değişkenlerin seçimi ve özeti
cat_cols = [col[0] for col in spark_df.dtypes if col[1] == 'string']

# ['surname', 'geography', 'gender']

spark_df.select("age").distinct().show()

distinct_geography = [x.geography for x in spark_df.select('geography').distinct().collect()]

# ['Germany', 'France', 'Spain']


# Churn'e göre sayısal değişkenlerin özet istatistikleri
for col in num_cols:
    spark_df.groupby("exited").agg({col: "mean"}).show()

# +------+-----------------+
# |exited|         avg(Age)|
# +------+-----------------+
# |     1| 44.8379970544919|
# |     0|37.40838879819164|
# +------+-----------------+

# üyenin aktif olup olmamasına göre churn ortalaması

spark_df.groupby("isactivemember").agg({'exited': "mean"}).show()
# +--------------+-------------------+
# |isactivemember|        avg(exited)|
# +--------------+-------------------+
# |             1|0.14269073966220153|
# |             0|0.26850897092183956|
# +--------------+-------------------+

#üyenin ülkesine göre churn olup olmama durumu
spark_df.groupby("geography").agg({'exited': "mean"}).show()
# +---------+-------------------+
# |geography|        avg(exited)|
# +---------+-------------------+
# |  Germany|0.32443204463929853|
# |   France|0.16154766653370561|
# |    Spain| 0.1667339523617279|
# +---------+-------------------+

from pyspark.sql import functions as F
spark_df.sort("creditscore").show() # küçükten büyüğe sıralar
spark_df.sort(F.desc("creditscore")).show() #büyükten küçüğe sıralar


##################################################
# DATA PREPROCESSING & FEATURE ENGINEERING
##################################################

##################################################
# Missing Value Handling
##################################################

from pyspark.sql.functions import when, count, col

spark_df.select([count(when(col(c).isNull(), c)).alias(c) for c in spark_df.columns]).toPandas().T

# rownumber        0
# customerid       0
# surname          0
# creditscore      0
# geography        0
# gender           0
# age              0
# tenure           0
# balance          0
# numofproducts    0
# hascrcard        0
# isactivemember   0
# estimatedsalary  0
# exited           0


##################################################
# Bucketization / Bining / Num to Cat
##################################################
############################
# Bucketizer ile Değişken Türetmek/Dönüştürmek
############################

from pyspark.ml.feature import Bucketizer

spark_df.select('age').describe().toPandas().transpose()

# summary  count     mean              stddev  min  max
# age      10000  38.9218  10.487806451704587   18   92


bucketizer = Bucketizer(splits=[0, 25, 35, 45, 65, 92], inputCol="age", outputCol="age_cat")
spark_df = bucketizer.setHandleInvalid("keep").transform(spark_df)
spark_df = spark_df.withColumn('age_cat', spark_df.age_cat + 1)

bucketizer_2 = Bucketizer(splits=[350,500,650,850], inputCol="creditscore", outputCol="cred_score_cat")
spark_df = bucketizer_2.setHandleInvalid("keep").transform(spark_df)
spark_df = spark_df.withColumn('cred_score_cat', spark_df.cred_score_cat + 1)


spark_df.groupby("age_cat").count().show()
# +-------+-----+
# |age_cat|count|
# +-------+-----+
# |    1.0|  457|
# |    4.0| 2058|
# |    3.0| 3981|
# |    2.0| 3222|
# |    5.0|  282|
# +-------+-----+
spark_df.groupby("age_cat").agg({'exited': "mean"}).show()
# |age_cat|        avg(exited)|
# +-------+-------------------+
# |    1.0|  0.087527352297593|
# |    4.0|0.48639455782312924|
# |    3.0|0.17658879678472744|
# |    2.0|0.07759155803848541|
# |    5.0| 0.1524822695035461|
# +-------+-------------------+

spark_df.groupby("cred_score_cat").count().show()
# +--------------+-----+
# |cred_score_cat|count|
# +--------------+-----+
# |           1.0|  632|
# |           3.0| 5100|
# |           2.0| 4268|
# +--------------+-----+
spark_df.groupby("cred_score_cat").agg({'exited': "mean"}).show()
# +--------------+-------------------+
# |cred_score_cat|        avg(exited)|
# +--------------+-------------------+
# |           1.0|0.23734177215189872|
# |           3.0|0.19372549019607843|
# |           2.0|0.21063730084348642|
# +--------------+-------------------+

spark_df = spark_df.withColumn("age_cat", spark_df["age_cat"].cast("integer")) #data type changed to int
spark_df = spark_df.withColumn("cred_score_cat", spark_df["cred_score_cat"].cast("integer")) #data type changed to int
spark_df.show(5)
# +---------+----------+---------+-----------+---------+------+---+------+---------+-------------+---------+--------------+---------------+------+-------+--------------+
# |rownumber|customerid|  surname|creditscore|geography|gender|age|tenure|  balance|numofproducts|hascrcard|isactivemember|estimatedsalary|exited|age_cat|cred_score_cat|
# +---------+----------+---------+-----------+---------+------+---+------+---------+-------------+---------+--------------+---------------+------+-------+--------------+
# |        1|  15634602| Hargrave|        619|   France|Female| 42|     2|      0.0|            1|        1|             1|      101348.88|     1|      3|             2|
# |        2|  15647311|     Hill|        608|    Spain|Female| 41|     1| 83807.86|            1|        0|             1|      112542.58|     0|      3|             2|
# |        3|  15619304|     Onio|        502|   France|Female| 42|     8| 159660.8|            3|        1|             0|      113931.57|     1|      3|             2|
# |        4|  15701354|     Boni|        699|   France|Female| 39|     1|      0.0|            2|        0|             0|       93826.63|     0|      3|             3|
# |        5|  15737888| Mitchell|        850|    Spain|Female| 43|     2|125510.82|            1|        1|             1|        79084.1|     0|      3|             3|



##################################################
# Encoding
##################################################

# Label Encoding: gender

indexer = StringIndexer(inputCol="gender", outputCol="gender_label")
indexer.fit(spark_df).transform(spark_df).show(5)

temp_sdf = indexer.fit(spark_df).transform(spark_df)
spark_df = temp_sdf.withColumn("gender_label", temp_sdf["gender_label"].cast("integer"))

spark_df = spark_df.drop('gender')


# Ordinal Encoding: geography_cat (France:1, Spain:2, Germany:3)
spark_df = spark_df.withColumn('geography_cat',
                               when(spark_df['geography'] == 'France', 1).
                               when(spark_df['geography'] == 'Spain', 2).otherwise(3))
spark_df = spark_df.drop('geography')
spark_df.show(5)
# +---------+----------+--------+-----------+---+------+---------+-------------+---------+--------------+---------------+------+-------+--------------+------------+-------------+
# |rownumber|customerid| surname|creditscore|age|tenure|  balance|numofproducts|hascrcard|isactivemember|estimatedsalary|exited|age_cat|cred_score_cat|gender_label|geography_cat|
# +---------+----------+--------+-----------+---+------+---------+-------------+---------+--------------+---------------+------+-------+--------------+------------+-------------+
# |        1|  15634602|Hargrave|        619| 42|     2|      0.0|            1|        1|             1|      101348.88|     1|      3|             2|           1|            1|
# |        2|  15647311|    Hill|        608| 41|     1| 83807.86|            1|        0|             1|      112542.58|     0|      3|             2|           1|            2|
# |        3|  15619304|    Onio|        502| 42|     8| 159660.8|            3|        1|             0|      113931.57|     1|      3|             2|           1|            1|
# |        4|  15701354|    Boni|        699| 39|     1|      0.0|            2|        0|             0|       93826.63|     0|      3|             3|           1|            1|
# |        5|  15737888|Mitchell|        850| 43|     2|125510.82|            1|        1|             1|        79084.1|     0|      3|             3|           1|            2|
# +---------+----------+--------+-----------+---+------+---------+-------------+---------+--------------+---------------+------+-------+--------------+------------+-------------+


##################################################
# TARGET'ın Tanımlanması
##################################################

# TARGET'ın tanımlanması
stringIndexer = StringIndexer(inputCol='exited', outputCol='label')
temp_sdf = stringIndexer.fit(spark_df).transform(spark_df)
spark_df = temp_sdf.withColumn("label", temp_sdf["label"].cast("integer"))


##################################################
# Feature'ların Tanımlanması
##################################################


cols = ['age', 'creditscore', 'tenure', 'balance', 'numofproducts', 'hascrcard',
        'isactivemember', 'estimatedsalary','age_cat','cred_score_cat','gender_label','geography_cat']


va = VectorAssembler(inputCols=cols, outputCol="features")
va_df = va.transform(spark_df)

final_df = va_df.select("features", "label")
final_df.show(5)

# +--------------------+-----+
# |            features|label|
# +--------------------+-----+
# |[42.0,619.0,2.0,0...|    1|
# |[41.0,608.0,1.0,8...|    0|
# |[42.0,502.0,8.0,1...|    1|
# |[39.0,699.0,1.0,0...|    0|
# |[43.0,850.0,2.0,1...|    0|
# +--------------------+-----+



# StandardScaler
# scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
# final_df = scaler.fit(final_df).transform(final_df)

##################################################
# TRAIN-TEST SPLIT
##################################################

train_df, test_df = final_df.randomSplit([0.8, 0.2], seed=17)

print("Training Dataset Count: " + str(train_df.count()))
# Training Dataset Count: 7952
print("Test Dataset Count: " + str(test_df.count()))
# Test Dataset Count: 2048


train_df.groupby('label').count().show()
# +-----+-----+
# |label|count|
# +-----+-----+
# |    1| 1601|
# |    0| 6351|
# +-----+-----+


# Oversampling is used to increase the observations with the label 1
# link: https://medium.com/@junwan01/oversampling-and-undersampling-with-pyspark-5dbc25cdf253
major_df = train_df.filter(train_df.label == 0)
minor_df = train_df.filter(train_df.label == 1)
ratio = int(major_df.count()/minor_df.count())
a = range(ratio)

oversampled_df = minor_df.withColumn('dummy', explode(array([lit(x) for x in a]))).drop('dummy')
train_new = major_df.unionAll(oversampled_df)
train_new.groupby('label').count().show()
# |label|count|
# +-----+-----+
# |    1| 4803|
# |    0| 6351|
# +-----+-----+



##################################################
# Logistic Regression
##################################################

log_model = LogisticRegression(featuresCol='features', labelCol='label').fit(train_new)
y_pred = log_model.transform(test_df)
y_pred.show()

y_pred.select("label", "prediction").show()
y_pred.filter(y_pred.label == y_pred.prediction).count() / y_pred.count()

evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="prediction", metricName='areaUnderROC')
evaluatorMulti = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")


acc = evaluatorMulti.evaluate(y_pred, {evaluatorMulti.metricName: "accuracy"})
precision = evaluatorMulti.evaluate(y_pred, {evaluatorMulti.metricName: "precisionByLabel"})
recall = evaluatorMulti.evaluate(y_pred, {evaluatorMulti.metricName: "recallByLabel"})
f1 = evaluatorMulti.evaluate(y_pred, {evaluatorMulti.metricName: "f1"})
roc_auc = evaluator.evaluate(y_pred)

print("accuracy: %f, precision: %f, recall: %f, f1: %f, roc_auc: %f" % (acc, precision, recall, f1, roc_auc))

# accuracy: 0.750488, precision: 0.878871, recall: 0.792184, f1: 0.763256, roc_auc: 0.694257

# Classification report :
y_true = y_pred.select('label').toPandas()
y_hat = y_pred.select('prediction').toPandas()

print(classification_report(y_true, y_hat))

#               precision    recall  f1-score   support
#            0       0.88      0.79      0.83      1612
#            1       0.44      0.60      0.50       436
#     accuracy                           0.75      2048
#    macro avg       0.66      0.69      0.67      2048
# weighted avg       0.78      0.75      0.76      2048

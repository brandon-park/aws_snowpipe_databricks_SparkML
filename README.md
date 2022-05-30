```python

```

# <center> Streaming Data Analysis 
# <center> With AWS S3, Snowflake, Databricks, and SparkMLlib

<center> <img src="https://github.com/brandon-park/aws_snowpipe_databricks_SparkML/blob/main/architecture.PNG?raw=true" width="100%"/>

## TOC:

1. [Introduction](#Introduction)
2. [Snowpipe configuration for streaming data](#snowpipe)
3. [Import data to Databricks](#import)
4. [Preprocessing](#prep)
5. [Modeling, Prediction, and Evaluation](#model)

## Introduction <a name="Introduction"></a>

### Streaming data into data wareshouse

In this project, we assume that the streaming data is being created and stored in AWS S3. Once S3 bucket has a new file, then it will send the notification to Snowpipe. Snowpipe is Snowflake's severless function that automatically detects the data and append to the existing table. 
To leverage parallel processing, we will connect the table in Snowflake to Spark Dataframe in Databricks. Lastly, Spark MLlib is used to predict the label. 

_disclaimer:
    The goal of this notebook is to showcase the data pipeline for streaming data. Hyper parameters used in each model are not optimized and hence the best model / performance are not literally 'the best' for this toy dataset._

## Snowpipe configuration for streaming data <a name="snowpipe"></a>


```python
%sql

// Below SQL query is to be run in Snowflake

create or replace storage integration s3_snowpipe
  TYPE = EXTERNAL_STAGE
  STORAGE_PROVIDER = S3
  ENABLED = TRUE 
  STORAGE_AWS_ROLE_ARN = 'arn:aws:iam::919247622774:role/brandon_snowpipe'
  STORAGE_ALLOWED_LOCATIONS = ('s3://wdbc/');
  
  
     
// See storage integration properties to fetch external_id so we can update it in S3
DESC integration s3_snowpipe;


CREATE OR REPLACE TABLE PROJECT_DB.PROJECT_TABLES.wdbc (
 num INT,
  id INT,	
  diagnosis INT,	
  mean_radius INT,	
  mean_texture INT,	
  mean_perimeter INT,	
  mean_area INT,	
  mean_smoothness INT,	
  mean_compactness INT,	
  mean_concavity INT,	
  mean_concave_points INT,	
  mean_symmetry INT,	
  mean_fractal_dimension INT,	
  se_radius	 INT,
  se_texture INT,	
  se_perimeter INT,	
  se_area INT,	
  se_smoothness INT,	
  se_compactness INT,	
  se_concavity INT,	
  se_concave_points INT,	
  se_symmetry INT,	
  se_fractal_dimension INT,	
  largest_radius INT,	
  largest_texture INT,	
  largest_perimeter INT,	
  largest_area INT,	
  largest_smoothness INT,	
  largest_compactness INT,	
  largest_concavity	 INT,
  largest_concave_points INT,	
  largest_symmetry	 INT,
  largest_fractal_dimension INT);


// Create file format object
CREATE OR REPLACE file format PROJECT_DB.file_formats.csv_fileformat
    type = csv
    field_delimiter = ','
    skip_header = 1
    null_if = ('NULL','null')
    empty_field_as_null = TRUE;
    
    
 // Create stage object with integration object & file format object
CREATE OR REPLACE stage PROJECT_DB.AWS_stages.wdbc_folder
    URL = 's3://wdbc/'
    STORAGE_INTEGRATION = s3_snowpipe
    FILE_FORMAT = PROJECT_DB.file_formats.csv_fileformat;
   

 // Create stage object with integration object & file format object
LIST @PROJECT_DB.AWS_stages.wdbc_folder;


// Create schema to keep things organized
CREATE OR REPLACE SCHEMA PROJECT_DB.pipes;

// Define pipe
CREATE OR REPLACE pipe PROJECT_DB.pipes.wdbc_pipe 
auto_ingest = TRUE
AS
COPY INTO PROJECT_DB.PROJECT_TABLES.wdbc
FROM @PROJECT_DB.AWS_stages.wdbc_folder
file_format= PROJECT_DB.file_formats.csv_fileformat;

// Describe pipe
DESC pipe wdbc_pipe;

SELECT * FROM PROJECT_DB.PROJECT_TABLES.wdbc;

```

## Import data to Databricks <a name="import"></a>


```python


from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import VectorIndexer, VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

import matplotlib.pyplot as plt
import pandas as pd 
```


```python
# snowflake connection options
options = {
  "sfUrl": "fs90326.us-east-2.aws.snowflakecomputing.com",
  "sfUser": user,
  "sfPassword": password,
  "sfDatabase": "PROJECT_DB",
  "sfSchema": "PROJECT_TABLES",
  "sfWarehouse": "COMPUTE_WH"
}
```


```python
# import data from Snowflake (Snowpipe)
wdbc = spark.read \
  .format("snowflake") \
  .options(**options) \
  .option("dbtable", "wdbc") \
  .load()
 
wdbc.show(n=5, truncate=True, vertical=False)
```


    +---+------+---------+-----------+------------+--------------+---------+---------------+----------------+--------------+-------------------+-------------+----------------------+---------+----------+------------+-------+-------------+--------------+------------+-----------------+-----------+--------------------+--------------+---------------+-----------------+------------+------------------+-------------------+-----------------+----------------------+----------------+-------------------------+
    |NUM|    ID|DIAGNOSIS|MEAN_RADIUS|MEAN_TEXTURE|MEAN_PERIMETER|MEAN_AREA|MEAN_SMOOTHNESS|MEAN_COMPACTNESS|MEAN_CONCAVITY|MEAN_CONCAVE_POINTS|MEAN_SYMMETRY|MEAN_FRACTAL_DIMENSION|SE_RADIUS|SE_TEXTURE|SE_PERIMETER|SE_AREA|SE_SMOOTHNESS|SE_COMPACTNESS|SE_CONCAVITY|SE_CONCAVE_POINTS|SE_SYMMETRY|SE_FRACTAL_DIMENSION|LARGEST_RADIUS|LARGEST_TEXTURE|LARGEST_PERIMETER|LARGEST_AREA|LARGEST_SMOOTHNESS|LARGEST_COMPACTNESS|LARGEST_CONCAVITY|LARGEST_CONCAVE_POINTS|LARGEST_SYMMETRY|LARGEST_FRACTAL_DIMENSION|
    +---+------+---------+-----------+------------+--------------+---------+---------------+----------------+--------------+-------------------+-------------+----------------------+---------+----------+------------+-------+-------------+--------------+------------+-----------------+-----------+--------------------+--------------+---------------+-----------------+------------+------------------+-------------------+-----------------+----------------------+----------------+-------------------------+
    |200|877501|        0|         12|          20|            79|      461|              0|               0|             0|                  0|            0|                     0|        0|         1|           2|     27|            0|             0|           0|                0|          0|                   0|            14|             28|               92|         638|                 0|                  0|                0|                     0|               0|                        0|
    |201|877989|        1|         18|          19|           115|      952|              0|               0|             0|                  0|            0|                     0|        0|         1|           3|     41|            0|             0|           0|                0|          0|                   0|            20|             26|              140|        1239|                 0|                  0|                0|                     0|               0|                        0|
    |202|878796|        1|         23|          27|           159|     1685|              0|               0|             0|                  0|            0|                     0|        1|         2|           5|     83|            0|             0|           0|                0|          0|                   0|            25|             33|              177|        1986|                 0|                  0|                1|                     0|               0|                        0|
    |203| 87880|        1|         14|          24|            92|      598|              0|               0|             0|                  0|            0|                     0|        1|         2|           4|     53|            0|             0|           0|                0|          0|                   0|            19|             42|              129|        1153|                 0|                  1|                0|                     0|               0|                        0|
    |204| 87930|        0|         12|          19|            81|      482|              0|               0|             0|                  0|            0|                     0|        0|         1|           2|     30|            0|             0|           0|                0|          0|                   0|            15|             25|               96|         678|                 0|                  0|                0|                     0|               0|                        0|
    +---+------+---------+-----------+------------+--------------+---------+---------------+----------------+--------------+-------------------+-------------+----------------------+---------+----------+------------+-------+-------------+--------------+------------+-----------------+-----------+--------------------+--------------+---------------+-----------------+------------+------------------+-------------------+-----------------+----------------------+----------------+-------------------------+
    only showing top 5 rows
    
    


## Preprocessing <a name="prep"></a>


```python
# Drop unnecessary columns
wdbc = wdbc.orderBy(("NUM")).drop('NUM','ID')
print((wdbc.count(), len(wdbc.columns)))
```


    (569, 31)
    



```python
# Feature engineering for Spark MLlib

target = 'DIAGNOSIS'
features = wdbc.schema.names
features.remove(target)

va = VectorAssembler(inputCols=features, outputCol='features')

va_df = va.transform(wdbc)
va_df = va_df.select(['features', target])
va_df.show(3)
```


    +--------------------+---------+
    |            features|DIAGNOSIS|
    +--------------------+---------+
    |(30,[0,1,2,3,10,1...|        1|
    |(30,[0,1,2,3,10,1...|        1|
    |(30,[0,1,2,3,10,1...|        1|
    +--------------------+---------+
    only showing top 3 rows
    
    


## Modeling, Prediction, and Evaluation <a name="model"></a>


```python
# Train/test split
(train, test) = va_df.randomSplit([0.7, 0.3])


# Modeling and prediction
gbt = GBTClassifier(featuresCol='features', labelCol=target, maxIter=10)
gbtmodel = gbt.fit(train)
pred = gbtmodel.transform(test)
pred.show(3)
```


    +--------------------+---------+--------------------+--------------------+----------+
    |            features|DIAGNOSIS|       rawPrediction|         probability|prediction|
    +--------------------+---------+--------------------+--------------------+----------+
    |(30,[0,1,2,3,10,1...|        0|[1.31731406833674...|[0.93305720958827...|       0.0|
    |(30,[0,1,2,3,10,1...|        0|[1.31731406833674...|[0.93305720958827...|       0.0|
    |(30,[0,1,2,3,10,1...|        0|[1.31731406833674...|[0.93305720958827...|       0.0|
    +--------------------+---------+--------------------+--------------------+----------+
    only showing top 3 rows
    
    



```python
# Model evaluation

evaluator = MulticlassClassificationEvaluator(
    labelCol=target, predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(pred)

print("Accuracy: ", round(accuracy,2))
```


    Accuracy:  0.93
    


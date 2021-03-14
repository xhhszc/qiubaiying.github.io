---
layout:     post
title:      tfrecords的使用
subtitle:   Get start tfrecords
date:       2021-03-14
author:     xhhszc
catalog: true
tags:
    - tensorflow
    - tfrecords
---

# tfrecords的使用
------

## 1. 生成tfrecords的数据
利用python写生成数据的脚本文件gen_data.py:
```python
# file of gen_data.py
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
conf = SparkConf()
spark = SparkSession.builder.config(conf=conf).getOrCreate()

# create data
data = [('James','','Smith','1991-04-01','M',3000),
  ('Michael','Rose','','2000-05-19','M',4000),
  ('Robert','','Williams','1978-09-05','M',4000),
  ('Maria','Anne','Jones','1967-12-01','F',4000),
  ('Jen','Mary','Brown','1980-02-17','F',-1)
]
columns = ["firstname","middlename","lastname","dob","gender","salary"]

# make a dataframe from data
# 注意：tfrecords不能接收二维数组，或string列表（如["abc", "sss"]），若dataframe中含有这些数据，则必须将dataframe的每列都转换为float或string或float数组。
df = spark.createDataFrame(data=data, schema=columns)

"""
+---------+----------+--------+----------+------+------+
|firstname|middlename|lastname|dob       |gender|salary|
+---------+----------+--------+----------+------+------+
|James    |          |Smith   |1991-04-01|M     |3000  |
|Michael  |Rose      |        |2000-05-19|M     |4000  |
|Robert   |          |Williams|1978-09-05|M     |4000  |
|Maria    |Anne      |Jones   |1967-12-01|F     |4000  |
|Jen      |Mary      |Brown   |1980-02-17|F     |-1    |
+---------+----------+--------+----------+------+------+
"""

# 将dataframe存储到hdfs路径中
your_path = "/user/test/"
df.repartition(10).write.mode('overwrite').format("tfrecords").option("recordType", "Example").save(your_path)

```
使用shell命令执行上述脚本文件：
```shell
spark-submit \
--driver-memory 20g \
--executor-cores 4 \
--executor-memory 11g \
--conf spark.dynamicAllocation.minExecutors=100 \
--conf spark.dynamicAllocation.maxExecutors=150 \
--conf spark.defualt.parallelism=1200 \
--conf spark.executor.memoryOverhead=3096 \
--queue "your_queue_name" \
--jars spark-connector_2.11-1.10.0.jar \
gen_data.py

```
其中jar包[spark-connector_2.11-1.10.0.jar](https://github.com/tensorflow/ecosystem/tree/master/spark/spark-tensorflow-connector)用于将dataframe存为tfrecords格式。


## 2. 读取tfrecords的数据

```python
import tensorflow as tf

def _parse_function(example_proto):
    features = {"firstname": tf.io.VarLenFeature(tf.string),
                "middlename": tf.io.VarLenFeature(tf.string),
                "lastname": tf.io.VarLenFeature(tf.string),
                "dob": tf.io.VarLenFeature(tf.string),
                "gender":tf.io.VarLenFeature(tf.string),
                "salary":tf.io.FixedLenFeature((1), tf.float32)}
                # tf.io.FixedLenFeature(shape, type), shape可以为二维，例如(3,2)
    parsed_feature = tf.io.parse_single_example(example_proto, features)
    return parsed_feature['firstname'], parsed_feature['middlename'], parsed_feature['lastname'], parsed_feature['dob'], parsed_feature['gender'], parsed_feature['salary']

def parse_dataset(data_file_path):
    num_threads = tf.data.experimental.AUTOTUNE
    if num_threads > 4:
        num_threads = 4 # 限制线程数量
    files_name = data_file_path + "/*"
    data_files = tf.data.Dataset.list_files(files_name)
    dataset = data_files.interleave(tf.data.TFRecordDataset, cycle_length=num_threads)
    dataset = dataset.shuffle(batch_size*10) #随机打乱数据
    dataset = data.repeat(epoches) #将数据重复epoches次，用于训练模型epoches次
    data_parsed = dataset.map(_parse_function, num_parallel_calls=num_threads)
    data_parsed = data_parsed.batch(batch_size)
    data_parsed = data_parsed.prefetch(1)
    data_parsed = data_parsed.make_one_shot_iterator()
    return data_parsed


def get_dataset():
    iterator = parse_dataset(data_file_path="/user/test/")
    try:
        while True:
            sample = sess.run(iterator)
            # sample[0] is firstname, ..., sample[5] is salary.
    except tf.errors.OutOfRangeError:
        print("end of the dataset")


get_dataset()
```

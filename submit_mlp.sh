#!/bin/bash
workers=8
iterations=100
dataset=ip_class.svm
trainS=0.8
cores=6

spark-submit --driver-memory 13000m --executor-memory 15000m --conf spark.driver.maxResultSize=4g \
	--executor-cores $cores --master local[*] --class scaladl.Classifier ./target/scaladl-1.0.0.jar  $dataset $trainS $iterations $workers $cores



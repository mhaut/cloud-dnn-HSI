/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package scaladl

import java.io._

import org.apache.spark.ml.feature.MinMaxScaler
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.scaladl.{Classifier}
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.joda.time.{DateTime, DateTimeZone}

object Classifier {

  def main(args: Array[String]): Unit = {
    if(args.length != 5){
       println("Dataset, train size, iterations, num_workers and num_cores should be pased.")
       return
    }
    val spark = SparkSession.builder
      .appName("ML Classifier")
      .getOrCreate()
    val dataset = "./dataset/" + args(0)
    val data = spark.read.format("libsvm").load(dataset)
    val scaler = new MinMaxScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")
    // materialize data lazily persisted in memory
    val scalerModel = scaler.fit(data)
    // rescale each feature to range [min, max].
    val scaledData = scalerModel.transform(data)
    val processedData = scaledData.select("label", "scaledFeatures").toDF("label", "features")
    val trainSize = args(1).toFloat
    val testSize = 1 - trainSize
    val split = processedData.randomSplit(Array(trainSize, testSize), 1234L)
    val train = split(0)
    val test = split(1)
    data.unpersist()
    val layers = Array[Int](200, 128, 64, 32, 17)
    // create the trainer and set its parameters
    val trainer = new Classifier()
      .setLayers(layers)
      .setBlockSize(256)
      .setSeed(DateTime.now(DateTimeZone.UTC).getMillis())
      .setMaxIter(args(2).toInt)
    // train the model
    println("Training on %d workers with %d cores each".format(args(3).toInt, args(4).toInt))
    val model = trainer.fit(train)
    println("Evaluating...")
    val result = model.transform(test)
    val predictionAndLabels = result.select("prediction", "label")
    val evaluator = new MulticlassClassificationEvaluator()
      .setMetricName("accuracy")

     println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")
  }
}

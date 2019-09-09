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

package org.apache.spark.ml.scaladl

import java.io._

import scala.collection.JavaConverters._
import scala.collection.mutable

import breeze.linalg.{sum => Bsum, DenseMatrix => BDM, DenseVector => BDV}
import org.apache.spark.ml.linalg.{DenseVector, Vector, Vectors}
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.sql.functions._

/**
 * Evaluates regression
 */
class RegressionEvaluator {

  def datasetToMatrix(df : DataFrame): BDM[Double] = {
     val hashes = mutable.ArrayBuilder.make[DenseVector]
     val values = mutable.ArrayBuilder.make[Double]
     val rows : Array[Row] = df.collect
     val x = df.count.toInt
     rows.map( r =>
       hashes += r.get(0).asInstanceOf[DenseVector])
     val arr = hashes.result()
     val y = arr(0).size.toInt
     arr.map(row => row.toArray.foreach(values += _))
     new BDM(x, y, values.result())
  }

  def evaluate(dataset : DataFrame): Double = {
    val x = datasetToMatrix(dataset.select("features"))
    val y = datasetToMatrix(dataset.select("prediction"))
    Bsum((y - x) :^ 2d) / (x.rows * x.cols)
  }
}

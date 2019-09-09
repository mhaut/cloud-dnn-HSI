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

import breeze.linalg.{sum => Bsum, CSCMatrix, DenseMatrix => BDM, DenseVector => BDV}
import org.apache.hadoop.fs.Path
import org.apache.spark.annotation.Since
import org.apache.spark.ml.{PredictionModel, Predictor, PredictorParams}
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.{DenseVector, Vector, Vectors}
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared.{HasMaxIter, HasSeed, HasStepSize, HasTol}
import org.apache.spark.ml.util._
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.sql.functions._

/** Params for Classifier. */
private[scaladl] trait ClassifierParams extends PredictorParams
  with HasSeed with HasMaxIter with HasTol with HasStepSize {
  /**
   * Layer sizes including input size and output size.
   *
   * @group param
   */
  @Since("1.5.0")
  final val layers: IntArrayParam = new IntArrayParam(this, "layers",
    "Sizes of layers from input layer to output layer. " +
      "E.g., Array(780, 100, 10) means 780 inputs, " +
      "one hidden layer with 100 neurons and output layer of 10 neurons.",
    (t: Array[Int]) => t.forall(ParamValidators.gt(0)) && t.length > 1)

  /** @group getParam */
  @Since("1.5.0")
  final def getLayers: Array[Int] = $(layers)

  /**
   * Block size for stacking input data in matrices to speed up the computation.
   * Data is stacked within partitions. If block size is more than remaining data in
   * a partition then it is adjusted to the size of this data.
   * Recommended size is between 10 and 1000.
   * Default: 128
   *
   * @group expertParam
   */
  @Since("1.5.0")
  final val blockSize: IntParam = new IntParam(this, "blockSize",
    "Block size for stacking input data in matrices. Data is stacked within partitions." +
      " If block size is more than remaining data in a partition then " +
      "it is adjusted to the size of this data. Recommended size is between 10 and 1000",
    ParamValidators.gt(0))

  /** @group expertGetParam */
  @Since("1.5.0")
  final def getBlockSize: Int = $(blockSize)

  /**
   * The solver algorithm for optimization.
   * Supported options: "gd" (minibatch gradient descent) or "l-bfgs".
   * Default: "l-bfgs"
   *
   * @group expertParam
   */
  @Since("2.0.0")
  final val solver: Param[String] = new Param[String](this, "solver",
    "The solver algorithm for optimization. Supported options: " +
      s"${Classifier.supportedSolvers.mkString(", ")}. (Default l-bfgs)",
    ParamValidators.inArray[String](Classifier.supportedSolvers))

  /** @group expertGetParam */
  @Since("2.0.0")
  final def getSolver: String = $(solver)

  /**
   * The initial weights of the model.
   *
   * @group expertParam
   */
  @Since("2.0.0")
  final val initialWeights: Param[Vector] = new Param[Vector](this, "initialWeights",
    "The initial weights of the model")

  /** @group expertGetParam */
  @Since("2.0.0")
  final def getInitialWeights: Vector = $(initialWeights)

  setDefault(maxIter -> 100, tol -> Double.MinPositiveValue, blockSize -> 128,
    solver -> Classifier.LBFGS, stepSize -> 0.03)
}

/**
 * Classifier implementation.
 * Each layer has ReLU activation function, output layer has linear activation.
 * Number of inputs has to be equal to the size of feature vectors and to number of outputs
 */
@Since("1.5.0")
class Classifier @Since("1.5.0") (
    @Since("1.5.0") override val uid: String)
  extends Predictor[Vector, Classifier, ClassifierModel]
  with ClassifierParams with DefaultParamsWritable {

  @Since("1.5.0")
  def this() = this(Identifiable.randomUID("ae"))

  /**
   * Sets the value of param [[layers]].
   *
   * @group setParam
   */
  @Since("1.5.0")
  def setLayers(value: Array[Int]): this.type = set(layers, value)

  /**
   * Sets the value of param [[blockSize]].
   * Default is 128.
   *
   * @group expertSetParam
   */
  @Since("1.5.0")
  def setBlockSize(value: Int): this.type = set(blockSize, value)

  /**
   * Sets the value of param [[solver]].
   * Default is "l-bfgs".
   *
   * @group expertSetParam
   */
  @Since("2.0.0")
  def setSolver(value: String): this.type = set(solver, value)

  /**
   * Set the maximum number of iterations.
   * Default is 100.
   *
   * @group setParam
   */
  @Since("1.5.0")
  def setMaxIter(value: Int): this.type = set(maxIter, value)

  /**
   * Set the convergence tolerance of iterations.
   * Smaller value will lead to higher accuracy with the cost of more iterations.
   * Default is 1E-6.
   *
   * @group setParam
   */
  @Since("1.5.0")
  def setTol(value: Double): this.type = set(tol, value)

  /**
   * Set the seed for weights initialization if weights are not set
   *
   * @group setParam
   */
  @Since("1.5.0")
  def setSeed(value: Long): this.type = set(seed, value)

  /**
   * Sets the value of param [[initialWeights]].
   *
   * @group expertSetParam
   */
  @Since("2.0.0")
  def setInitialWeights(value: Vector): this.type = set(initialWeights, value)

  /**
   * Sets the value of param [[stepSize]] (applicable only for solver "gd").
   * Default is 0.03.
   *
   * @group setParam
   */
  @Since("2.0.0")
  def setStepSize(value: Double): this.type = set(stepSize, value)

  @Since("1.5.0")
  override def copy(extra: ParamMap): Classifier = defaultCopy(extra)

  /**
   * Overloaded train method, allows printing output to a file.
   *
   * @param dataset Training dataset
   * @param pw File write buffer
   * @return Fitted model
   */
  override protected def train(dataset: Dataset[_]): ClassifierModel = {
    val myLayers = $(layers)
    val labels = myLayers.last
    val lpData = extractLabeledPoints(dataset)
    val data = lpData.map(lp => LabelConverter.encodeLabeledPoint(lp, labels))
    val topology = FeedForwardTopology.classifier(myLayers)
    val trainer = new FeedForwardTrainer(topology, myLayers(0), myLayers.last)
    if (isDefined(initialWeights)) {
      trainer.setWeights($(initialWeights))
    } else {
      trainer.setSeed($(seed))
    }
    if ($(solver) == Classifier.LBFGS) {
      trainer.LBFGSOptimizer
        .setNumIterations($(maxIter))
    } else if ($(solver) == Classifier.GD) {
      trainer.SGDOptimizer
        .setNumIterations($(maxIter))
        .setConvergenceTol($(tol))
        .setStepSize($(stepSize))
    } else {
      throw new IllegalArgumentException(
        s"The solver $solver is not supported by Classifier.")
    }
    trainer.setStackSize($(blockSize))
    val classifier = trainer.train(data)
    new ClassifierModel(uid, myLayers, classifier.weights)
  }

  /**
  Necessary wrapper due to train method being protected.
  **/
  def startTrain(dataset: Dataset[_]): ClassifierModel = {
    val classifier = this.train(dataset)
    classifier
  }
}

@Since("2.0.0")
object Classifier
  extends DefaultParamsReadable[Classifier] {

  /** String name for "l-bfgs" solver. */
  private[scaladl] val LBFGS = "l-bfgs"

  /** String name for "gd" (minibatch gradient descent) solver. */
  private[scaladl] val GD = "gd"

  /** Set of solvers that Classifier supports. */
  private[scaladl] val supportedSolvers = Array(LBFGS, GD)

  @Since("2.0.0")
  override def load(path: String): Classifier = super.load(path)
}

/**
 *  model based on the Classifier.
 * Each layer has ReLU activation function, output layer has empty.
 *
 * @param uid uid
 * @param layers array of layer sizes including input and output layers
 * @param weights the weights of layers
 */
@Since("1.5.0")
class ClassifierModel private[ml] (
    @Since("1.5.0") override val uid: String,
    @Since("1.5.0") val layers: Array[Int],
    @Since("2.0.0") val weights: Vector)
  extends PredictionModel[Vector, ClassifierModel]
  with Serializable with MLWritable {

  @Since("1.6.0")
  override val numFeatures: Int = layers.head

  private val classifier = FeedForwardTopology
    .classifier(layers)
    .model(weights)

  /**
   * Returns layers in a Java List.
   */
  private[ml] def javaLayers: java.util.List[Int] = {
    layers.toList.asJava
  }

  /**
   * Predict label for the given features.
   * This internal method is used to implement [[transform()]] and output [[predictionCol]].
   */
  override protected def predict(features: Vector): Double = {
    LabelConverter.decodeLabel(classifier.predict(features))
  }

  @Since("1.5.0")
  override def copy(extra: ParamMap): ClassifierModel = {
    copyValues(new ClassifierModel(uid, layers, weights), extra)
  }

  @Since("2.0.0")
  override def write: MLWriter =
    new 	ClassifierModel.ClassifierModelWriter(this)
}

@Since("2.0.0")
object ClassifierModel
  extends MLReadable[ClassifierModel] {

  @Since("2.0.0")
  override def read: MLReader[ClassifierModel] =
    new ClassifierModelReader

  @Since("2.0.0")
  override def load(path: String): ClassifierModel = super.load(path)

  /** [[MLWriter]] instance for [[ClassifierModel]] */
  private[ClassifierModel]
  class ClassifierModelWriter(
      instance: ClassifierModel) extends MLWriter {

    private case class Data(layers: Array[Int], weights: Vector)

    override protected def saveImpl(path: String): Unit = {
      // Save metadata and Params
      DefaultParamsWriter.saveMetadata(instance, path, sc)
      // Save model data: layers, weights
      val data = Data(instance.layers, instance.weights)
      val dataPath = new Path(path, "data").toString
      sparkSession.createDataFrame(Seq(data)).repartition(1).write.parquet(dataPath)
    }
  }

  private class ClassifierModelReader
    extends MLReader[ClassifierModel] {

    /** Checked against metadata when loading model */
    private val className = classOf[ClassifierModel].getName

    override def load(path: String): ClassifierModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc, className)

      val dataPath = new Path(path, "data").toString
      val data = sparkSession.read.parquet(dataPath).select("layers", "weights").head()
      val layers = data.getAs[Seq[Int]](0).toArray
      val weights = data.getAs[Vector](1)
      val model = new ClassifierModel(metadata.uid, layers, weights)

      DefaultParamsReader.getAndSetParams(model, metadata)
      model
    }
  }
}

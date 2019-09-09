# Cloud Deep Networks for Hyperspectral Image Analysis
The Code for "Cloud Deep Networks for Hyperspectral Image Analysis". [https://ieeexplore.ieee.org/document/8798981]
```
J. M. Haut, Jose A. Gallardo, M. E. Paoletti, G. Cavallaro, J. Plaza, A. Plaza and M. Riedel.
Cloud Deep Networks for Hyperspectral Image Analysis. 
IEEE Transactions on Geoscience and Remote Sensing
DOI: 10.1109/TGRS.2019.2929731 
Accepted for publication, 2019.
```

![Pipeline](https://github.com/mhaut/cloud-dnn-HSI/blob/master/images/pipeline.png)

For other hand, this code is based on **A Scalable Implementation of Deep Learning on Spark** [https://github.com/avulanov/scalable-deeplearning] and the implementation of artificial neural networks in **Spark ML** (https://spark.apache.org/docs/latest/ml-classification-regression.html#multilayer-perceptron-classifier). In addition to the multilayer perceptron, it contains new deep learning features such as Autoencoders, new activations functions and a cleaned up output results presentation.

## Installation
### Requirements
  - Apache Spark 2.1.1 or higher
  - Java and Scala
  - Maven

### Build 
Clone and compile:
```
git clone https://github.com/mhaut/cloud-dnn-HSI.git
cd cloud-dnn-HSI
mvn clean install
```
The jar library will be availabe in `target` folder. After compiling, a ready-to-go example of use is provided before.

## Example of use
### Built-in examples
Scaladl provides working examples of Indian Pines classification and regression. Examples are in **scaladl** [https://github.com/mhaut/cloud-dnn-hsi/tree/master/src/main/scala/scaladl] package. They can be run via Spark submit or by running
the provided bash scripts:

#### Classification
```
./spark-submit --class scaladl.Classifier --master spark://master:7077 /path/to/scaldl.jar name_of_data_svm train_size iterations workers cores
```
or
```
./submit_mlp.sh
```

#### Regression
```
./spark-submit --class scaladl.Autoencoder --master spark://master:7077 /path/to/scaldl.jar name_of_data_svm train_size iterations workers cores
```
or
```
./submit_ae.sh
```

#### Important note
In order for this to work, datasets should be placed on dataset folder in the root of the project, provided datasets are placed on that folder. If you want to test the suite on other dataset, tools to convert it to svm are provided over the utils/ folder.


### Spark shell
Start Spark with this library:
```
./spark-shell --jars scaladl.jar
```
Or use it as external dependency for your application.

## Models Pipeline
### Multilayer perceptron
Indian Pines classification example
  - Load [Indian Pines data](http://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes) stored in LibSVM format as a DataFrame
  - Initialize the multilayer perceptron classifier with a input layer, a set of hidden layers and the output layer
  - Train and predict

Full pipeline can be consulted on the provided [https://github.com/mhaut/cloud-dnn-HSI/blob/master/src/main/scala/scaladl/Classifier.scala]
### Autoencoder
Indian Pines regression example
  - Load Indian Pines data, without removed bands stored in LibSVM format as a DataFrame
  - Initialize the autoencoder with a input layer, a set of tied hidden layers and the output layer
  - Train and predict autoencoder

Full pipeline can be consulted on the provided [https://github.com/mhaut/cloud-dnn-HSI/blob/master/src/main/scala/scaladl/Autoencoder.scala]

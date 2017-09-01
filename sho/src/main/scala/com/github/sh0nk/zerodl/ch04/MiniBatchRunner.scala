package com.github.sh0nk.zerodl.ch04

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.stats.distributions.Rand
import com.github.sh0nk.zerodl.ch03.MNISTLoader

import scala.reflect.ClassTag

class MiniBatchRunner(loader: MNISTLoader) {
  val hiddenLayer = 50
  val batchSize = 100
  val iterNum = 10000
  val learningRate = 0.1

  var trainX, testX: DenseMatrix[Double] = _
  var trainY, testY: DenseVector[Int] = _ // Internal t in NN is 10 class, but this Y is not one hot
  var network: NumericalGradientNN = _

  private def loadData() = {
    println("start file loading")
    val trainData = loader.loadImage("/var/folders/6t/dlnd80jj05s8r8qcjmgm4q5cv9pfnn/T/zerodl-mnist/train-images-idx3-ubyte.gz")
    val trainLabel = loader.loadLabel("/var/folders/6t/dlnd80jj05s8r8qcjmgm4q5cv9pfnn/T/zerodl-mnist/train-labels-idx1-ubyte.gz", oneHot = false)
    val testData = loader.loadImage("/var/folders/6t/dlnd80jj05s8r8qcjmgm4q5cv9pfnn/T/zerodl-mnist/t10k-images-idx3-ubyte.gz")
    val testLabel = loader.loadLabel("/var/folders/6t/dlnd80jj05s8r8qcjmgm4q5cv9pfnn/T/zerodl-mnist/t10k-labels-idx1-ubyte.gz", oneHot = false)

    trainX = MiniBatchRunner.toDenseMatrixFromNestedArray(trainData)
    trainY = MiniBatchRunner.toDenseMatrixFromNestedArray(trainLabel).toDenseVector
    testX = MiniBatchRunner.toDenseMatrixFromNestedArray(testData)
    testY = MiniBatchRunner.toDenseMatrixFromNestedArray(testLabel).toDenseVector
    println(s"Train (${trainX.rows}, ${trainX.cols}) (${trainY.length})")
    println(s"Test (${testX.rows}, ${testX.cols}) (${testY.length})")
  }

  def run(): Unit = {
    loadData()
    network = new NumericalGradientNN(trainX.cols, hiddenLayer, trainY.length)

    Range(0, iterNum).foreach { v =>
      println(s"batch attempt ${v}")
      batch()
    }
  }

  def batch(): Unit = {
    var batchX = DenseMatrix.zeros[Double](batchSize, trainX.cols)
//    println(s"batchX ${batchX.rows}, ${batchX.cols}")
    var batchY = DenseVector.zeros[Int](batchSize)
    val randIdx = Rand.permutation(trainX.rows).get().slice(0, batchSize)
//    println(randIdx)

    randIdx.zipWithIndex.foreach { case (v, i) =>
      batchX(i, ::) := trainX(v, ::)
      batchY(i) = trainY(v)
    }

    val diffW = network.numericalGradient(batchX, batchY)
    network.w -= (diffW * learningRate)
    val loss = network.loss(batchX, batchY)
    println(loss)
  }
}

object MiniBatchRunner {
  def toDenseMatrixFromNestedArray[T:ClassTag](array: Array[Array[T]]): DenseMatrix[T] = {
    val rows = array.length
    if (rows == 0) {
      return new DenseMatrix[T](0, 0)
    }
    val cols = array(0).length

    new DenseMatrix[T](rows, cols, array.flatten)
  }

  def testMatConverting(): Unit = {
    val act = toDenseMatrixFromNestedArray(Array(Array(2, 3, 5), Array(1, 4, 6)))
    if (act.toArray sameElements Array(2, 3, 5, 1, 4, 6)) {
      println("same")
    }
  }

  def main(args: Array[String]) = {
//    testMatConverting()

    val runner = new MiniBatchRunner(new MNISTLoader())
    runner.run()
  }
}

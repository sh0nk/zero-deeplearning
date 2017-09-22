package com.github.sh0nk.zerodl.ch04

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.stats.distributions.Rand
import com.github.sh0nk.zerodl.ch03.{Downloader, MNISTLoader}

import scala.reflect.ClassTag

class MiniBatchRunner(loader: MNISTLoader) {
  val hiddenLayer = 50
  val outputLayer = 10

  val batchSize = 100
  val iterNum = 10000
  val learningRate = 0.1

  var trainX, testX: DenseMatrix[Double] = _
  var trainY, testY: DenseVector[Int] = _ // Internal t in NN is 10 class, but this Y is not one hot
  var network: NumericalGradientNN = _

  private def loadData() = {
    Logger.info("start file loading")
    val dir = s"${Downloader.baseDir}/"
    val trainData = loader.loadImage(dir + Downloader.keyFile("train_img"))
    val trainLabel = loader.loadLabel(dir + Downloader.keyFile("train_label"), oneHot = false)
    val testData = loader.loadImage(dir + Downloader.keyFile("test_img"))
    val testLabel = loader.loadLabel(dir + Downloader.keyFile("test_label"), oneHot = false)

    trainX = MiniBatchRunner.toDenseMatrixFromNestedArray(trainData)
    trainY = MiniBatchRunner.toDenseMatrixFromNestedArray(trainLabel).toDenseVector
    testX = MiniBatchRunner.toDenseMatrixFromNestedArray(testData)
    testY = MiniBatchRunner.toDenseMatrixFromNestedArray(testLabel).toDenseVector
    Logger.info(s"Train (${trainX.rows}, ${trainX.cols}) (${trainY.length})")
    Logger.info(s"Test (${testX.rows}, ${testX.cols}) (${testY.length})")
  }

  def run(): Unit = {
    loadData()
    network = new NumericalGradientNN(trainX.cols, hiddenLayer, outputLayer)

    Range(0, iterNum).foreach { v =>
      Logger.info(s"batch attempt ${v}")
      batch()
    }
  }

  def batch(): Unit = {
    var batchX = DenseMatrix.zeros[Double](batchSize, trainX.cols)
    Logger.debug(s"batchX ${batchX.rows}, ${batchX.cols}")
    var batchY = DenseVector.zeros[Int](batchSize)
    val randIdx = Rand.permutation(trainX.rows).get().slice(0, batchSize)
    Logger.debug(randIdx)

    randIdx.zipWithIndex.foreach { case (v, i) =>
      batchX(i, ::) := trainX(v, ::)
      batchY(i) = trainY(v)
    }

    val diffW = network.numericalGradient(batchX, batchY)
    Logger.debug(s"d.W1 ${diffW.W1}")
    network.w -= (diffW * learningRate)
    val loss = network.loss(batchX, batchY)
    Logger.info(s"loss val: $loss")
  }
}

object MiniBatchRunner {
  def toDenseMatrixFromNestedArray[T:ClassTag](array: Array[Array[T]]): DenseMatrix[T] = {
    val rows = array.length
    if (rows == 0) {
      return new DenseMatrix[T](0, 0)
    }
    val cols = array(0).length

    // The array is row-major order (DenseMatrix default is column-major)
    new DenseMatrix[T](rows, cols, array.flatten, 0, cols, true)
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

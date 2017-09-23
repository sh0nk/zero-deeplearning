package com.github.sh0nk.zerodl.ch05

import breeze.linalg.{*, DenseMatrix, argmax, sum}
import breeze.numerics.abs
import breeze.stats.distributions.Rand
import com.github.sh0nk.zerodl.ch03.{Downloader, MNISTLoader}
import com.github.sh0nk.zerodl.ch04.{Logger, MiniBatchRunner, NumericalGradientNN}
import com.github.sh0nk.zerodl.ch06.{BackpropagationGradientNNch06, WeightLayer}

class BGMiniBatchRunnerch06(loader: MNISTLoader) {
  val hiddenLayer = 50
  val outputLayer = 10

  val batchSize = 100
  val iterNum = 10000
  val learningRate = 0.1

  var trainX, testX: DenseMatrix[Double] = _
  var trainY, testY: DenseMatrix[Int] = _
  var network: BackpropagationGradientNNch06 = _

  private def loadData() = {
    Logger.info("start file loading")
    Downloader.ensureDownloadingAll()
    val dir = s"${Downloader.baseDir}/"
    val trainData = loader.loadImage(dir + Downloader.keyFile("train_img"))
    val trainLabel = loader.loadLabel(dir + Downloader.keyFile("train_label"), oneHot = true)
    val testData = loader.loadImage(dir + Downloader.keyFile("test_img"))
    val testLabel = loader.loadLabel(dir + Downloader.keyFile("test_label"), oneHot = true)

    trainX = MiniBatchRunner.toDenseMatrixFromNestedArray(trainData)
    trainY = MiniBatchRunner.toDenseMatrixFromNestedArray(trainLabel)
    testX = MiniBatchRunner.toDenseMatrixFromNestedArray(testData)
    testY = MiniBatchRunner.toDenseMatrixFromNestedArray(testLabel)
    Logger.info(s"Train (${trainX.rows}, ${trainX.cols}) (${trainY.rows}, ${trainY.cols})")
    Logger.info(s"Test (${testX.rows}, ${testX.cols}) (${testY.rows}, ${testY.cols})")
  }

  def run(): Unit = {
    loadData()
    network = new BackpropagationGradientNNch06(trainX.cols, hiddenLayer, outputLayer)

    Range(0, iterNum).foreach { v =>
      Logger.info(s"batch attempt ${v}")
      batch()
    }
  }

  def batch(): Unit = {
    var batchX = DenseMatrix.zeros[Double](batchSize, trainX.cols)
    Logger.debug(s"batchX ${batchX.rows}, ${batchX.cols}")
    var batchY = DenseMatrix.zeros[Int](batchSize, trainY.cols)
    val randIdx = Rand.permutation(trainX.rows).get().slice(0, batchSize)
    Logger.debug(randIdx)

    randIdx.zipWithIndex.foreach { case (v, i) =>
      batchX(i, ::) := trainX(v, ::)
      batchY(i, ::) := trainY(v, ::)
    }

    gradientDescent(batchX, batchY)

    val loss = network.loss(batchX, batchY)
    Logger.info(s"loss val: $loss")
  }

  def gradientDescent(batchX: DenseMatrix[Double], batchY: DenseMatrix[Int]): Unit = {
    network.gradient(batchX, batchY)

    Logger.debug(s"d.W1 ${network.layers.head.asInstanceOf[WeightLayer].dW("w")}")

    network.layers.filter(_.isInstanceOf[WeightLayer]).map(_.asInstanceOf[WeightLayer]).foreach { layer =>
      layer.W.keys.map { key =>
        Logger.trace(s"gradientDescent key for W: $key")
        layer.W(key) -= layer.dW(key) * learningRate
      }
      layer.Wb.keys.map { key =>
        Logger.trace(s"gradientDescent key for Wb: $key")
        layer.Wb(key) -= layer.dWb(key) * learningRate
      }
    }
  }
}

object BGMiniBatchRunnerch06 {
  def testGradientNumericalGradientDiff(): Unit = {
    val runner = new BGMiniBatchRunnerch06(new MNISTLoader())
    runner.loadData()

    val network = new BackpropagationGradientNNch06(runner.trainX.cols, runner.hiddenLayer, runner.outputLayer)
    val refNetwork = new NumericalGradientNN(runner.trainX.cols, runner.hiddenLayer, runner.outputLayer)

    val idx = 0 to 2
    network.gradient(runner.trainX(idx, ::), runner.trainY(idx, ::))
    val bgGradW1 = network.layers.head.asInstanceOf[WeightLayer].dW("w")
    val yMat = runner.trainY(idx, ::)
    val numGrad = refNetwork.numericalGradient(runner.trainX(idx, ::), argmax(yMat(*, ::)))
    Logger.info(yMat)
    Logger.info(argmax(yMat(*, ::)))
    //    Logger.info(bgGrad.W1)
    //    Logger.info(numGrad.W1)

    Logger.info(s"W1 ${sum(abs(bgGradW1 - numGrad.W1)) / runner.trainX.cols / runner.hiddenLayer}")
  }

  def main(args: Array[String]) = {
//    testGradientNumericalGradientDiff()

    val runner = new BGMiniBatchRunnerch06(new MNISTLoader())
    runner.run()
  }
}

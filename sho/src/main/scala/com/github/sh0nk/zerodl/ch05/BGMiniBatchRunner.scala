package com.github.sh0nk.zerodl.ch05

import breeze.linalg.{*, DenseMatrix, argmax, sum}
import breeze.numerics.abs
import breeze.stats.distributions.Rand
import com.github.sh0nk.matplotlib4j.Plot
import com.github.sh0nk.zerodl.ch03.{Downloader, MNISTLoader}
import com.github.sh0nk.zerodl.ch04.{Logger, MiniBatchRunner, NumericalGradientNN}

class BGMiniBatchRunner(loader: MNISTLoader) {
  val hiddenLayer = 50
  val outputLayer = 10

  val batchSize = 100
  val iterNum = 10000
  val learningRate = 0.1

  var trainX, testX: DenseMatrix[Double] = _
  var trainY, testY: DenseMatrix[Int] = _
  var network: BackpropagationGradientNN = _

  var accTrain, accTest: Seq[Double] = Seq()
  var dispLosses: Seq[Double] = Seq()

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
    network = new BackpropagationGradientNN(trainX.cols, hiddenLayer, outputLayer)

    Range(0, iterNum).foreach { v =>
      Logger.info(s"batch attempt ${v}")
      batch()
      if (v % 100 == 0) {
        collectAccuracy()
        if (v % 2000 == 0) {
          drawLoss()
          drawAccuracy()
        }
      }
    }
    drawAccuracy()
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

    val diffW = network.gradient(batchX, batchY)
    Logger.debug(s"d.W1 ${diffW.W1}")
    network.w -= (diffW * learningRate)
    val loss = network.loss(batchX, batchY)
    dispLosses :+= loss
    Logger.info(s"loss val: $loss")
  }

  def collectAccuracy(): Unit = {
    accTrain :+= network.accuracy(trainX, trainY)
    accTest :+= network.accuracy(testX, testY)
  }

  def drawAccuracy() {
    import scala.collection.JavaConverters._

    val plt = Plot.create()
    plt.title("SGD Accuracy")
    plt.plot().add(accTrain.indices.map(Int.box).toList.asJava, accTrain.map(Double.box).toList.asJava)
      .linestyle("-").label("Train")
    plt.plot().add(accTest.indices.map(Int.box).toList.asJava, accTest.map(Double.box).toList.asJava)
      .linestyle("--").label("Test")
    plt.legend().loc("upper right")
    plt.show()
  }

  def drawLoss() {
    import scala.collection.JavaConverters._

    val plt = Plot.create()
    plt.title("SGD Loss")
    plt.plot().add(dispLosses.indices.map(Int.box).toList.asJava, dispLosses.map(Double.box).toList.asJava)
      .linestyle("-").label("Loss")
    plt.legend().loc("upper right")
    plt.show()
  }
}

object BGMiniBatchRunner {
  def testGradientNumericalGradientDiff(): Unit = {
    val runner = new BGMiniBatchRunner(new MNISTLoader())
    runner.loadData()

    val network = new BackpropagationGradientNN(runner.trainX.cols, runner.hiddenLayer, runner.outputLayer)
    val refNetwork = new NumericalGradientNN(runner.trainX.cols, runner.hiddenLayer, runner.outputLayer)

    val idx = 0 to 2
    val bgGrad = network.gradient(runner.trainX(idx, ::), runner.trainY(idx, ::))
    val yMat = runner.trainY(idx, ::)
    val numGrad = refNetwork.numericalGradient(runner.trainX(idx, ::), argmax(yMat(*, ::)))
    Logger.info(yMat)
    Logger.info(argmax(yMat(*, ::)))
//    Logger.info(bgGrad.W1)
//    Logger.info(numGrad.W1)

    Logger.info(s"W1 ${sum(abs(bgGrad.W1 - numGrad.W1)) / runner.trainX.cols / runner.hiddenLayer}")
  }

  def main(args: Array[String]) = {
//    testGradientNumericalGradientDiff()

    val runner = new BGMiniBatchRunner(new MNISTLoader())
    runner.run()
  }
}

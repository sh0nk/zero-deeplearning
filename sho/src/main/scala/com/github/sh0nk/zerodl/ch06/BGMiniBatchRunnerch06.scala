package com.github.sh0nk.zerodl.ch05

import breeze.linalg.{*, DenseMatrix, argmax, sum}
import breeze.numerics.abs
import breeze.stats.distributions.Rand
import com.github.sh0nk.matplotlib4j.Plot
import com.github.sh0nk.zerodl.ch03.{Downloader, MNISTLoader}
import com.github.sh0nk.zerodl.ch04.{Logger, MiniBatchRunner, NumericalGradientNN}
import com.github.sh0nk.zerodl.ch06._

import scala.collection.mutable.ArrayBuffer

class BGMiniBatchRunnerch06(loader: MNISTLoader) {
  val batchSize = 100

  var trainX, testX: DenseMatrix[Double] = _
  var trainY, testY: DenseMatrix[Int] = _

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

  def drawLoss(dispLossesMulti: Seq[Seq[Double]]) {
    import scala.collection.JavaConverters._

    val plt = Plot.create()
    plt.title("SGD Loss")
    dispLossesMulti.foreach { dispLosses =>
      plt.plot().add(dispLosses.indices.map(Int.box).toList.asJava, dispLosses.map(Double.box).toList.asJava)
        .linestyle("-").label("Loss")
    }
    plt.legend().loc("upper right")
    plt.show()
  }

  // FIXME: Merge to drawLoss
  def drawAccuracy(dispAccuracyMulti: Seq[Seq[Double]]) {
    import scala.collection.JavaConverters._

    val plt = Plot.create()
    plt.title("SGD Accuracy")
    dispAccuracyMulti.foreach { dispLosses =>
      plt.plot().add(dispLosses.indices.map(Int.box).toList.asJava, dispLosses.map(Double.box).toList.asJava)
        .linestyle("-").label("Accuracy")
    }
    plt.legend().loc("upper right")
    plt.show()
  }

  def batch(network: BackpropagationGradientNNch06, optimizer: Optimizer = SGD(0.01)): Double = {
    var batchX = DenseMatrix.zeros[Double](batchSize, trainX.cols)
    Logger.debug(s"batchX ${batchX.rows}, ${batchX.cols}")
    var batchY = DenseMatrix.zeros[Int](batchSize, trainY.cols)
    val randIdx = Rand.permutation(trainX.rows).get().slice(0, batchSize)
    Logger.debug(randIdx)

    randIdx.zipWithIndex.foreach { case (v, i) =>
      batchX(i, ::) := trainX(v, ::)
      batchY(i, ::) := trainY(v, ::)
    }

    gradientDescent(network, batchX, batchY, optimizer)

    val loss = network.loss(batchX, batchY)
    Logger.info(s"loss val: $loss")

    loss
  }

  def accuracy(network: BackpropagationGradientNNch06): Double = {
    network.accuracy(trainX, trainY)
  }

  private def gradientDescent(network: BackpropagationGradientNNch06,
                      batchX: DenseMatrix[Double], batchY: DenseMatrix[Int], optimizer: Optimizer): Unit = {
    network.gradient(batchX, batchY)

    Logger.debug(s"d.W1 ${network.layers.head.asInstanceOf[WeightLayer].dW("w")}")

    optimizer.optimize(network.layers)
  }
}

object BGMiniBatchRunnerch06 {
  val hiddenLayer = 50
  val outputLayer = 10
  val iterNum = 10000

  def testGradientNumericalGradientDiff(): Unit = {
    val runner = new BGMiniBatchRunnerch06(new MNISTLoader())
    runner.loadData()

    val network = new BackpropagationGradientNNch06(runner.trainX.cols, Seq(hiddenLayer), outputLayer)
    val refNetwork = new NumericalGradientNN(runner.trainX.cols, hiddenLayer, outputLayer)

    val idx = 0 to 2
    network.gradient(runner.trainX(idx, ::), runner.trainY(idx, ::))
    val bgGradW1 = network.layers.head.asInstanceOf[WeightLayer].dW("w")
    val yMat = runner.trainY(idx, ::)
    val numGrad = refNetwork.numericalGradient(runner.trainX(idx, ::), argmax(yMat(*, ::)))
//    Logger.info(yMat)
//    Logger.info(argmax(yMat(*, ::)))
    //    Logger.info(bgGrad.W1)
    //    Logger.info(numGrad.W1)

    Logger.info(s"W1 ${sum(abs(bgGradW1 - numGrad.W1)) / runner.trainX.cols / hiddenLayer}")
  }


  def testOneOptimizer(): Unit = {
    val runner = new BGMiniBatchRunnerch06(new MNISTLoader())

    runner.loadData()
    val optimizer: Optimizer = AdaGrad()
    val network = new BackpropagationGradientNNch06(runner.trainX.cols, Seq(hiddenLayer), outputLayer)
    var dispLosses: Seq[Double] = Seq()

    Range(0, iterNum).foreach { v =>
      Logger.info(s"batch attempt ${v}")
      dispLosses :+= runner.batch(network, optimizer)
      if (v % 2000 == 0) {
        runner.drawLoss(Seq(dispLosses))
      }
    }
  }

  def testOptimizersComparison(): Unit = {
    val runner = new BGMiniBatchRunnerch06(new MNISTLoader())

    runner.loadData()
    val networksAndOptimizersAndLosses = Seq(SGD(0.1), Momentum(), AdaGrad()).map { v =>
      (new BackpropagationGradientNNch06(runner.trainX.cols, Seq(hiddenLayer), outputLayer), v, ArrayBuffer[Double]())
    }

    Range(0, iterNum).foreach { v =>
      Logger.info(s"batch attempt ${v}")
      networksAndOptimizersAndLosses.foreach(nol => nol._3 += runner.batch(nol._1, nol._2))

      if (v % 2000 == 0) {
        runner.drawLoss(networksAndOptimizersAndLosses.map(_._3))
      }
    }
  }

  def testWeightInitDeboost(): Unit = {
    val runner = new BGMiniBatchRunnerch06(new MNISTLoader())

    runner.loadData()
    val networksAndLosses = Seq(
      (new BackpropagationGradientNNch06(runner.trainX.cols, Seq(100, 100, 100), outputLayer,
        weightInitDeboost = "0.01"), ArrayBuffer[Double](), ArrayBuffer[Double]()),
      (new BackpropagationGradientNNch06(runner.trainX.cols, Seq(100, 100, 100), outputLayer,
        weightInitDeboost = "xavier"), ArrayBuffer[Double](), ArrayBuffer[Double]()),
      (new BackpropagationGradientNNch06(runner.trainX.cols, Seq(100, 100, 100), outputLayer,
        weightInitDeboost = "he"), ArrayBuffer[Double](), ArrayBuffer[Double]())
    )

    Range(0, iterNum).foreach { v =>
      Logger.info(s"batch attempt ${v}")
      networksAndLosses.foreach(nl => nl._2 += runner.batch(nl._1))

      // Per Epoch
      if (v % (runner.trainX.rows / runner.batchSize) == 0) {
        networksAndLosses.foreach(nl => nl._3 += runner.accuracy(nl._1))
      }

      if (v % 2000 == 0) {
        runner.drawLoss(networksAndLosses.map(_._2))

        runner.drawAccuracy(networksAndLosses.map(_._3))
      }
    }
  }

  // FIXME: This method never converges... When `weightInitStd=0.2`, then no batch norm version converged
  // It means no batch norm version works as the book, so calculation of BatchNormalization
  // should have some problems
  def testBatchNormalization(): Unit = {
    val runner = new BGMiniBatchRunnerch06(new MNISTLoader())

    runner.loadData()
    // Reduce the number of data
    runner.trainX = runner.trainX(0 until 1000, ::)
    runner.trainY = runner.trainY(0 until 1000, ::)

    val networksAndLosses = Seq(
      (new BackpropagationGradientNNch06(runner.trainX.cols, Seq(100, 100, 100, 100, 100), outputLayer,
        weightInitDeboost = "0.01", weightInitStd = 0.05), ArrayBuffer[Double](), ArrayBuffer[Double]()),
      (new BackpropagationGradientNNch06(runner.trainX.cols, Seq(100, 100, 100, 100, 100), outputLayer,
        weightInitDeboost = "0.01", weightInitStd = 0.05, useBatchNorm = true), ArrayBuffer[Double](), ArrayBuffer[Double]())
    )

    Range(0, iterNum).foreach { v =>
      Logger.info(s"batch attempt ${v}")
      networksAndLosses.foreach(nl => nl._2 += runner.batch(nl._1))

      // Per Epoch
      if (v % (runner.trainX.rows / runner.batchSize) == 0) {
        networksAndLosses.foreach(nl => nl._3 += runner.accuracy(nl._1))
      }

      if (v % 1000 == 0 && v > 0) {
        runner.drawAccuracy(networksAndLosses.map(_._3))
//        runner.drawLoss(networksAndLosses.map(_._2))

        return
      }
    }
  }


  def main(args: Array[String]) = {
    //    testGradientNumericalGradientDiff()

//        testOptimizersComparison()
//        testWeightInitDeboost()
        testBatchNormalization()
  }
}

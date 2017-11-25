package com.github.sh0nk.zerodl.ch06

import breeze.linalg.{*, BitVector, DenseMatrix, DenseVector, argmax, randn, sum}
import com.github.sh0nk.zerodl.ch04.Logger
import com.github.sh0nk.zerodl.ch05.{Layer, OutputLayer, ReLU, SoftmaxWithLoss}

import scala.collection.mutable.ListBuffer

class BackpropagationGradientNNch06(inputSize: Int, hiddenSize: Seq[Int], outputSize: Int,
                                    weightInitStd: Double = 0.01, weightInitDeboost: String = "he",
                                    useBatchNorm: Boolean = false) {
  var layers: ListBuffer[Layer] = ListBuffer()
  var lastLayer: OutputLayer = _

  val weightInitDeboostFunc = weightInitDeboost match {
    case "he" => n: Double => math.sqrt(2.0 / n)
    case "xavier" => n: Double => math.sqrt(1.0 / n)
    case _ => n: Double => weightInitStd
  }

  initLayers()

  def initLayers(): Unit = {
    val multiSize = ListBuffer[Int]()
    multiSize += inputSize
    multiSize ++= hiddenSize
    multiSize += outputSize

    val multiLayers = (multiSize zip multiSize.drop(1)).foreach { case (first, second) =>
      layers += Affine(
        weightInitDeboostFunc(first) * randn((first, second)),
        DenseVector.zeros(second))

      if (useBatchNorm) {
        layers += BatchNormalization(DenseVector.ones[Double](second).t, DenseVector.zeros[Double](second).t)
      }

      layers += ReLU()
    }
    if (useBatchNorm) {
      layers = layers.dropRight(1) // remove last batch normalization, to make the last Affine
    }
    layers = layers.dropRight(1) // remove last activation func

    lastLayer = SoftmaxWithLoss()
  }

  def forward(x: DenseMatrix[Double], train: Boolean = false): DenseMatrix[Double] = {
    var newX = x
    Logger.trace(s"x ${x.rows}, ${x.cols}, " +
      s"W1 ${layers.head.asInstanceOf[WeightLayer].W("w").rows}, ${layers.head.asInstanceOf[WeightLayer].W("w").cols}, " +
      s"b1 ${layers.head.asInstanceOf[WeightLayer].Wb("b").length}")
    layers.foreach { l =>
      newX = l.forward(newX, train)
    }
    newX
  }

  def predict(x: DenseMatrix[Double], train: Boolean = false): DenseMatrix[Double] = forward(x, train)

  def loss(x: DenseMatrix[Double], t: DenseMatrix[Int], train: Boolean = false): Double = {
    val newX = predict(x, train)
    lastLayer.forward(newX, t)
  }

  def accuracy(x: DenseMatrix[Double], t: DenseMatrix[Int]): Double = {
    val newX = predict(x)
    val yIdx: DenseVector[Int] = argmax(newX(*, ::))
    val tIdx: DenseVector[Int] = argmax(t(*, ::))
    val correctMask: BitVector = yIdx :== tIdx
    val correctMaskInt: DenseVector[Int] = correctMask.map(if (_) 1 else 0)
    val corrects: Int = sum(correctMaskInt)

    corrects.toDouble / t.rows
  }

  def gradient(x: DenseMatrix[Double], t: DenseMatrix[Int]) = {
    loss(x, t, train = true)

    var dout = lastLayer.backward(1.0)

    layers.reverse.foreach { l =>
      dout = l.backward(dout)
    }
  }

}



object BackpropagationGradientNN {
  def main(args: Array[String]): Unit = {
//    test1()
//    testMat2Vec()
  }

}

package com.github.sh0nk.zerodl.ch06

import breeze.linalg.{DenseMatrix, DenseVector, randn}
import com.github.sh0nk.zerodl.ch04.Logger
import com.github.sh0nk.zerodl.ch05.{Layer, OutputLayer, ReLU, SoftmaxWithLoss}

import scala.collection.mutable.ListBuffer

class BackpropagationGradientNNch06(inputSize: Int, hiddenSize: Seq[Int], outputSize: Int,
                                    weightInitStd: Double = 0.01, weightInitDeboost: String = "he") {
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
      layers ++= Seq(Affine(
        weightInitDeboostFunc(first) * randn((first, second)),
        DenseVector.zeros(second)),
      ReLU())
    }
    layers = layers.dropRight(1) // remove last activation func

    lastLayer = SoftmaxWithLoss()
  }

  def forward(x: DenseMatrix[Double]): DenseMatrix[Double] = {
    var newX = x
    Logger.trace(s"x ${x.rows}, ${x.cols}, " +
      s"W1 ${layers.head.asInstanceOf[WeightLayer].W("w").rows}, ${layers.head.asInstanceOf[WeightLayer].W("w").cols}, " +
      s"b1 ${layers.head.asInstanceOf[WeightLayer].Wb("b").length}")
    layers.foreach { l =>
      newX = l.forward(newX)
    }
    newX
  }

  def predict(x: DenseMatrix[Double]): DenseMatrix[Double] = forward(x)

  def loss(x: DenseMatrix[Double], t: DenseMatrix[Int]): Double = {
    val newX = predict(x)
    lastLayer.forward(newX, t)
  }

  def gradient(x: DenseMatrix[Double], t: DenseMatrix[Int]) = {
    loss(x, t)

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

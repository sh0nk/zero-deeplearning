package com.github.sh0nk.zerodl.ch06

import breeze.linalg.{DenseMatrix, DenseVector, randn}
import com.github.sh0nk.zerodl.ch04.Logger
import com.github.sh0nk.zerodl.ch05.{Layer, OutputLayer, ReLU, SoftmaxWithLoss}

class BackpropagationGradientNNch06(inputSize: Int, hiddenSize: Int, outputSize: Int, weightInitStd: Double = 0.01) {
  var layers: Seq[Layer] = _
  var lastLayer: OutputLayer = _
  initLayers()

  def initLayers(): Unit = {
    layers = Seq(
      Affine(
        randn((inputSize, hiddenSize)) * weightInitStd,
        DenseVector.zeros(hiddenSize)),
      ReLU(),
      Affine(
        randn((hiddenSize, outputSize)) * weightInitStd,
        DenseVector.zeros(outputSize))
    )
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

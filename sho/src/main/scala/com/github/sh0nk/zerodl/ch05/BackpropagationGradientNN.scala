package com.github.sh0nk.zerodl.ch05

import breeze.linalg.{*, DenseMatrix, DenseVector, argmax, randn, sum}
import com.github.sh0nk.zerodl.ch04.Logger

class BackpropagationGradientNN(inputSize: Int, hiddenSize: Int, outputSize: Int, weightInitStd: Double = 0.01) {
  var w = new Weight()
  var layers: Seq[Layer] = _
  var lastLayer: OutputLayer = _
  initWeights()
  initLayers()

  def initWeights(): Unit = {
    Logger.info(s"inputSize ${inputSize}, hiddenSize ${hiddenSize}, outputSize ${outputSize}")
    w.W1 = randn((inputSize, hiddenSize)) * weightInitStd
    w.b1 = DenseVector.zeros(hiddenSize)
    w.W2 = randn((hiddenSize, outputSize)) * weightInitStd
    w.b2 = DenseVector.zeros(outputSize)
  }

  def initLayers(): Unit = {
    layers = Seq(
      Affine(w.W1, w.b1),
      ReLU(),
      Affine(w.W2, w.b2)
    )
    lastLayer = SoftmaxWithLoss()
  }

  def forward(x: DenseMatrix[Double]): DenseMatrix[Double] = {
    var newX = x
    Logger.trace(s"x ${x.rows}, ${x.cols}, W1 ${w.W1.rows}, ${w.W1.cols}, b1 ${w.b1.length}")
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

  def gradient(x: DenseMatrix[Double], t: DenseMatrix[Int]): Weight = {
    loss(x, t)

    var dout = lastLayer.backward(1.0)

    layers.reverse.foreach { l =>
      dout = l.backward(dout)
    }

    val grad = new Weight()
    grad.W1 = layers(0).asInstanceOf[Affine].dW
    grad.b1 = layers(0).asInstanceOf[Affine].db
    grad.W2 = layers(2).asInstanceOf[Affine].dW
    grad.b2 = layers(2).asInstanceOf[Affine].db

    grad
  }

  def accuracy(x: DenseMatrix[Double], t: DenseMatrix[Int]): Double = {
    val y = predict(x)
    val mask = argmax(y(*, ::)).map(v => if (v != 0) true else false) & argmax(t(*, ::)).map(v => if (v != 0) true else false)
    sum(mask.map(if(_) 1.0 else 0.0)) / x.rows
  }

}

// Replace with DenseVector extension for broadcast
class Weight {
  var W1, W2: DenseMatrix[Double] = _
  var b1, b2: DenseVector[Double] = _

  def *(v: Double): Weight = {
    val tmp = new Weight()
    tmp.W1 = v * W1
    tmp.W2 = v * W2
    tmp.b1 = v * b1
    tmp.b2 = v * b2
    tmp
  }

  def -=(d: Weight) = {
    W1 -= d.W1
    W2 -= d.W2
    b1 -= d.b1
    b2 -= d.b2
  }
}


object BackpropagationGradientNN {
  def main(args: Array[String]): Unit = {
//    test1()
//    testMat2Vec()
  }


}

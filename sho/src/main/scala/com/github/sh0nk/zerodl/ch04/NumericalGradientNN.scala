package com.github.sh0nk.zerodl.ch04

import breeze.linalg.{*, DenseMatrix, DenseVector, randn, randomDouble}
import com.github.sh0nk.zerodl.ch03.ActivationFunctions

class NumericalGradientNN(inputSize: Int, hiddenSize: Int, outputSize: Int, weightInitStd: Double = 0.01) {
  var w = new Weight()
  var outputFunction: DenseMatrix[Double] => DenseMatrix[Double] = MatrixActivationFunctions.softmax
  init()

  def init(): Unit = {
    Logger.info(s"inputSize ${inputSize}, hiddenSize ${hiddenSize}, outputSize ${outputSize}")
    w.W1 = randn((inputSize, hiddenSize)) * weightInitStd
    w.b1 = DenseVector.zeros(hiddenSize)
    w.W2 = randn((hiddenSize, outputSize)) * weightInitStd
    w.b2 = DenseVector.zeros(outputSize)
  }

  def forward(x: DenseMatrix[Double]): DenseMatrix[Double] = {
    Logger.trace(s"x ${x.rows}, ${x.cols}, W1 ${w.W1.rows}, ${w.W1.cols}, b1 ${w.b1.length}")
    var a1 = x * w.W1
    a1(*, ::) += w.b1
    val z1 = MatrixActivationFunctions.sigmoid(a1)
    var a2 = z1 * w.W2
    a2(*, ::) += w.b2
    Logger.trace(s"x ${x.rows}, ${x.cols}, W2 ${w.W2.rows}, ${w.W2.cols}, b2 ${w.b2.length} = a2 ${a2.rows} ${a2.cols}")
    outputFunction(a2)
  }

  def predict(x: DenseMatrix[Double]): DenseMatrix[Double] = forward(x)

  def loss(x: DenseMatrix[Double], t: DenseVector[Int]): Double = {
    val y = predict(x)
    MatrixLossFunctions.crossEntropyError(y, t)
  }

  def numericalGradient(x: DenseMatrix[Double], t: DenseVector[Int]): Weight = {
    def f(w: DenseMatrix[Double]): Double = loss(x, t)

    val grad = new Weight()
    grad.W1 = Gradients.numericalGradient(f, w.W1)
    grad.b1 = Gradients.numericalGradient(f, w.b1.toDenseMatrix).toDenseVector
    grad.W2 = Gradients.numericalGradient(f, w.W2)
    grad.b2 = Gradients.numericalGradient(f, w.b2.toDenseMatrix).toDenseVector

    grad
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


object NumericalGradientNN {
  def main(args: Array[String]): Unit = {
//    test1()
    testMat2Vec()
  }

  def test1(): Unit = {
    val vec = DenseVector(1, 3, 5).t
    val mat = new DenseMatrix(3, 2, Array(1, 1, 1, 1, 1, 1))
    println(vec * mat)

    val summer = DenseVector(1, 1).t
    println(vec * mat + summer)
  }

  def test2(): Unit = {
    val nn = new NumericalGradientNN(784, 100, 10)
////    val y = nn.forward(DenseVector(1.0, 0.5))
//    print(y)
  }

  def testMat2Vec() = {
    val mat = new DenseMatrix(1, 5, Array(1, 2, 3, 4, 5))
    if (mat.toDenseVector == mat(0, ::).inner) {
      println("equals")
    }
  }

}

package com.github.sh0nk.zerodl.ch03

import breeze.linalg.{DenseMatrix, DenseVector}

class TriNetwork {
  var W1, W2, W3: DenseMatrix[Double] = null
  var b1, b2, b3: DenseVector[Double] = null

  var outputFunction: DenseVector[Double] => DenseVector[Double] = ActivationFunctions.softmax

  def loadDummyWeights(): Unit = {
    W1 = DenseMatrix((0.1, 0.3, 0.5), (0.2, 0.4, 0.6))
    b1 = DenseVector(0.1, 0.2, 0.3)
    W2 = DenseMatrix((0.1, 0.4), (0.2, 0.5), (0.3, 0.6))
    b2 = DenseVector(0.1, 0.2)
    W3 = DenseMatrix((0.1, 0.3), (0.2, 0.4))
    b3 = DenseVector(0.1, 0.2)

    outputFunction = ActivationFunctions.identity
  }

  def forward(x: DenseVector[Double]): DenseVector[Double] = {
    val a1 = x.t * W1 + b1.t
    val z1 = ActivationFunctions.sigmoid(a1.inner)
    val a2 = z1.t * W2 + b2.t
    val z2 = ActivationFunctions.sigmoid(a2.inner)
    val a3 = z2.t * W3 + b3.t
    outputFunction(a3.inner)
  }
}

object SimpleNN {
  def main(args: Array[String]): Unit = {
//    test1()
    test2()
  }

  def test1(): Unit = {
    val vec = DenseVector(1, 3, 5).t
    val mat = new DenseMatrix(3, 2, Array(1, 1, 1, 1, 1, 1))
    println(vec * mat)

    val summer = DenseVector(1, 1).t
    println(vec * mat + summer)
  }

  def test2(): Unit = {
    val nn = new TriNetwork()
    nn.loadDummyWeights()
    val y = nn.forward(DenseVector(1.0, 0.5))
    print(y)
  }

}

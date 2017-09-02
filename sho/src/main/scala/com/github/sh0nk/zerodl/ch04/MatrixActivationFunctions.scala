package com.github.sh0nk.zerodl.ch04

import breeze.linalg.{*, DenseMatrix, DenseVector, max, sum}
import breeze.numerics._

object MatrixActivationFunctions {
  def step(x: DenseMatrix[Double]): DenseMatrix[Double] = {
    x.map(v => if (v > 0.0) 1.0 else 0.0)
  }

  def sigmoid(x: DenseMatrix[Double]): DenseMatrix[Double] = {
//    breeze.numerics.sigmoid(x)
    x.map(v => 1.0 / (1.0 + math.exp(-v)))
  }

  def identity(x: DenseMatrix[Double]): DenseMatrix[Double] = x

  def softmax(x: DenseMatrix[Double]): DenseMatrix[Double] = {
    Logger.trace(s"softmax x: ${x.rows}, ${x.cols}")
    val c = max(x(*, ::))
    val exped = exp(x - c * DenseVector.ones[Double](x.cols).t)
    val sum_exp = sum(exped(*, ::))
    exped / (sum_exp * DenseVector.ones[Double](x.cols).t)
  }

  def main(args: Array[String]): Unit = {
    println(MatrixActivationFunctions.step(new DenseMatrix[Double](1, Array[Double](2, 1, 0, -1, -2), 0)))
    println(MatrixActivationFunctions.softmax(new DenseMatrix[Double](1, Array[Double](2, 1, 0, -1, -2), 0)))
  }
}



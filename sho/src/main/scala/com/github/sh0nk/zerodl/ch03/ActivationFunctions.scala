package com.github.sh0nk.zerodl.ch03

import breeze.linalg.{DenseMatrix, DenseVector}
//import breeze.numerics._

object ActivationFunctions {
  def step(x: DenseVector[Double]): DenseVector[Double] = x.map(v => if (v > 0.0) 1.0 else 0.0)

  def sigmoid(x: DenseVector[Double]): DenseVector[Double] = {
    x.map(v => 1.0 / (1.0 + math.exp(-v)))
  }

  def identity(x: DenseVector[Double]): DenseVector[Double] = x

  def softmax(x: DenseVector[Double]): DenseVector[Double] = {
    val c = x.reduce((a, b) => math.max(a, b))
    val exp = x.map(v => math.exp(v - c))
    val sum_exp = x.reduce(_ + _)
    exp.map(_ / sum_exp)
  }

  def main(args: Array[String]): Unit = {
    println(ActivationFunctions.step(new DenseVector[Double](Array[Double](2, 1, 0, -1, -2))))
  }
}



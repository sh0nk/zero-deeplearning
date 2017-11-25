package com.github.sh0nk.zerodl.ch05

import breeze.linalg.DenseMatrix

trait Layer {
  def forward(x: DenseMatrix[Double], train: Boolean = false): DenseMatrix[Double]

  def backward(dout: DenseMatrix[Double]): DenseMatrix[Double]
}

trait OutputLayer {
  def forward(x: DenseMatrix[Double], t: DenseMatrix[Int]): Double

  def backward(dout: Double = 1): DenseMatrix[Double]
}

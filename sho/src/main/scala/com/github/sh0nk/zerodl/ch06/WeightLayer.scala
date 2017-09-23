package com.github.sh0nk.zerodl.ch06

import breeze.linalg.{DenseMatrix, DenseVector}
import com.github.sh0nk.zerodl.ch05.Layer


trait WeightLayer extends Layer {
  var W: Map[String, DenseMatrix[Double]] = Map()
  var Wb: Map[String, DenseVector[Double]] = Map()
  var dW: Map[String, DenseMatrix[Double]] = Map()
  var dWb: Map[String, DenseVector[Double]] = Map()

  def forward(x: DenseMatrix[Double]): DenseMatrix[Double]

  def backward(dout: DenseMatrix[Double]): DenseMatrix[Double]
}

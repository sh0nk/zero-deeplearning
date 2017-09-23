package com.github.sh0nk.zerodl.ch06

import breeze.linalg.{*, DenseMatrix, DenseVector, sum}

case class Affine(var initW: DenseMatrix[Double], var initb: DenseVector[Double]) extends WeightLayer {
  var x: DenseMatrix[Double] = _
  val wKey = "w"
  val bKey = "b"
  W = Map(wKey -> initW)
  Wb = Map(bKey -> initb)

  override def forward(x: DenseMatrix[Double]): DenseMatrix[Double] = {
    this.x = x
    var mul = x * W(wKey)
    //    mul(::, *) += b
    mul(*, ::) += Wb(bKey) // add row vec to all the rows
    mul
  }

  override def backward(dout: DenseMatrix[Double]): DenseMatrix[Double] = {
    val dX = dout * W(wKey).t
    dW += (wKey -> x.t * dout)
    dWb += (bKey -> sum(dout(::, *)).t)
    //    db = sum(dout(*, ::))
    dX
  }
}


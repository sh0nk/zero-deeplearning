package com.github.sh0nk.zerodl.ch05

import breeze.linalg.{*, DenseMatrix, DenseVector, sum}
import breeze.numerics.exp
import com.github.sh0nk.zerodl.ch04.{Logger, MatrixActivationFunctions, MatrixLossFunctions}

case class SoftmaxWithLoss() extends OutputWithLoss(MatrixActivationFunctions.softmax) {
  override def backward(dout: Double = 1): DenseMatrix[Double] = {
    val batchSize = t.rows
    val sub = y - t.map(_.toDouble)
    sub / batchSize.toDouble
  }
}

abstract class OutputWithLoss(outputActivation: (DenseMatrix[Double] => DenseMatrix[Double])) extends OutputLayer {
  var loss: Double = _
  var y: DenseMatrix[Double] = _
  var t: DenseMatrix[Int] = _

  override def forward(x: DenseMatrix[Double], t: DenseMatrix[Int]): Double = {
    this.t = t
    y = outputActivation(x)
    loss = MatrixLossFunctions.crossEntropyError(y, t)
    loss
  }
}

case class Affine(var W: DenseMatrix[Double], var b: DenseVector[Double]) extends Layer {
  var x: DenseMatrix[Double] = _
  var dW: DenseMatrix[Double] = _
  var db: DenseVector[Double] = _

  override def forward(x: DenseMatrix[Double]): DenseMatrix[Double] = {
    this.x = x
    var mul = x * W
    Logger.trace(mul.rows)
    Logger.trace(b.length)
//    mul(::, *) += b
    mul(*, ::) += b // add row vec to all the rows
    mul
  }

  override def backward(dout: DenseMatrix[Double]): DenseMatrix[Double] = {
    val dX = dout * W.t
    dW = x.t * dout
    db = sum(dout(::, *)).t // TODO?
//    db = sum(dout(*, ::))
    dX
  }
}

case class Sigmoid() extends Layer {
  var captured: DenseMatrix[Double] = _

  override def forward(x: DenseMatrix[Double]): DenseMatrix[Double] = {
    captured = DenseMatrix.ones[Double](x.rows, x.cols) /:/ (exp(-x) +:+ DenseMatrix.ones[Double](x.rows, x.cols))
    captured
  }

  override def backward(dout: DenseMatrix[Double]): DenseMatrix[Double] = {
    dout * (DenseMatrix.ones[Double](dout.rows, dout.cols) - captured) * captured
  }
}


case class ReLU() extends Layer {
  var maskIdx: DenseMatrix[Boolean] = _

  override def forward(x: DenseMatrix[Double]): DenseMatrix[Double] = {
    maskIdx = x <:= DenseMatrix.zeros[Double](x.rows, x.cols)
    mask(x)
  }

  override def backward(dout: DenseMatrix[Double]): DenseMatrix[Double] = {
    mask(dout)
  }

  private def mask(x: DenseMatrix[Double]): DenseMatrix[Double] = {
    maskIdx.foreachPair { case ((r, c), mask) =>
      if (mask) {
        x.update((r, c), 0)
      }
    }
    x
  }
}

object Layers {
    def main(args: Array[String]) = {
      if (DenseMatrix((1.0, 0.0), (0.0, 3.0)) != ReLU().forward(DenseMatrix((1.0, -2.0), (-1.0, 3.0)))) {
      throw new IllegalArgumentException("test failed")
    }
  }
}
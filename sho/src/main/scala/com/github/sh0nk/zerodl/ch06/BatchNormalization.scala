package com.github.sh0nk.zerodl.ch06

import breeze.linalg.{*, DenseMatrix, DenseVector, Transpose, sum}
import breeze.numerics._
import com.github.sh0nk.zerodl.ch05.Layer


// TODO: Support *Tensor* later
case class BatchNormalization(
                             var gamma: Transpose[DenseVector[Double]],
                             var beta: Transpose[DenseVector[Double]],
                             momentum: Double = 0.9,
                             var runningMean: Option[Transpose[DenseVector[Double]]] = None,
                             var runningVar: Option[Transpose[DenseVector[Double]]] = None,
                             ) extends Layer {
  var batchSize: Int = _
  var xc: DenseMatrix[Double] = _
  var xn: DenseMatrix[Double] = _
  var std: Transpose[DenseVector[Double]] = _
  var dGamma: Transpose[DenseVector[Double]] = _
  var dBeta: Transpose[DenseVector[Double]] = _
  var N, D: Int = _

  override def forward(x: DenseMatrix[Double], train: Boolean = false): DenseMatrix[Double] = {
//    var N, D: Int = 0
    if (runningMean.isEmpty) {
      println("init BatchNormalization only once per layer")
      N = x.rows
      D = x.cols
      runningMean = Some(DenseVector.zeros[Double](D).t)
      runningVar = Some(DenseVector.zeros[Double](D).t)
    }

    var tmpXn: DenseMatrix[Double] = x

    // Train only once before #backward is called
    // There is no easy way than having train flag on signature, because the internal layers cannot find
    // in which phase the forward is called only by the given x
    if (!train) {
      val tmpXc: DenseMatrix[Double] = x(*, ::) - runningMean.get.inner
      tmpXn = tmpXc(*, ::) / sqrt(runningVar.get.inner + 10e-7)

    } else {
      batchSize = x.rows

      val mu: Transpose[DenseVector[Double]] = sum(x(::, *)) / batchSize.toDouble
      xc = x(*, ::) - mu.inner

      val powed: DenseMatrix[Double] = xc *:* xc
      val variance: Transpose[DenseVector[Double]] = sum(powed(::, *)) / batchSize.toDouble
      std = sqrt(variance + 10e-7)
      xn = xc(*, ::) / std.inner

      runningMean = Some(runningMean.get * momentum + mu * (1.0 - momentum))
      runningVar = Some(runningVar.get * momentum + variance * (1.0 - momentum))
      tmpXn = xn
    }

    val gammaXn: DenseMatrix[Double] = tmpXn(*, ::) *:* gamma.inner
    gammaXn(*, ::) +:+ beta.inner
  }

  override def backward(dout: DenseMatrix[Double]): DenseMatrix[Double] = {
    dBeta = sum(dout(::, *))
    val tmpDGamma: DenseMatrix[Double] = xn *:* dout
    dGamma = sum(tmpDGamma(::, *))

    val dXn: DenseMatrix[Double] = dout(*, ::) *:* gamma.inner
    var dXc: DenseMatrix[Double] = dXn(*, ::) /:/ std.inner

    val tmpDStd1: DenseMatrix[Double] = dXn *:* xc
    val tmpDStd2: DenseMatrix[Double] = tmpDStd1(*, ::) /:/ (std *:* std).inner
    val dStd: Transpose[DenseVector[Double]] = sum(tmpDStd2(::, *)) * -1.0
    val dVar: Transpose[DenseVector[Double]] = (dStd /:/ std) * 0.5
    val tmpDXc = xc(*, ::) *:* dVar.inner
    tmpDXc :*= 2.0 / batchSize.toDouble
    dXc += tmpDXc
    val dMu = sum(dXc(::, *))

    // Feedback to beta and gamma internally (different from the book implementation)
    beta = dBeta
    gamma = dGamma

    val dX: DenseMatrix[Double] = dXc(*, ::) - (dMu.inner / batchSize.toDouble)
    dX
  }

}
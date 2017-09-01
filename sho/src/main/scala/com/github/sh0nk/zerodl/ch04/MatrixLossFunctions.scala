package com.github.sh0nk.zerodl.ch04

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics._

object MatrixLossFunctions {
  def crossEntropyError(y: DenseMatrix[Double], t: DenseVector[Int]): Double = {
    val batchSize = y.rows

    (0 until y.rows).map { r =>
      log(y(r, t(r)))
    }.sum / batchSize
  }

}

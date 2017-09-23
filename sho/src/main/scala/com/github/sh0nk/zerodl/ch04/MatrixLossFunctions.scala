package com.github.sh0nk.zerodl.ch04

import breeze.linalg.{*, DenseMatrix, DenseVector, argmax}
import breeze.numerics._

object MatrixLossFunctions {
  def crossEntropyError(y: DenseMatrix[Double], t: DenseVector[Int]): Double = {
    val batchSize = y.rows

    -1.0 * (0 until batchSize).map { r =>
      log(y(r, t(r)))
    }.sum / batchSize
  }

  def crossEntropyError(y: DenseMatrix[Double], t: DenseMatrix[Int]): Double = {
    crossEntropyError(y, ontHotToIndexVec(t))
  }

   def ontHotToIndexVec(t: DenseMatrix[Int]): DenseVector[Int] = {
    argmax(t(*, ::))
  }

  def main(args: Array[String]) {
    // Test
    if (DenseVector(1, 2) != ontHotToIndexVec(DenseMatrix((0, 1, 0), (0, 0, 1)))) {
      throw new IllegalArgumentException("test failed")
    }
  }

}

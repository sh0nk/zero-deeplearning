package com.github.sh0nk.zerodl.ch04

import breeze.linalg.DenseMatrix

object Gradients {

  private def numericalDiff(f: DenseMatrix[Double] => Double,
                            xHigh: DenseMatrix[Double], xLow: DenseMatrix[Double], delta: Double): Double = {
    (f(xHigh) - f(xLow)) / (2 * delta)
  }

  def numericalGradient(f: DenseMatrix[Double] => Double, x: DenseMatrix[Double]): DenseMatrix[Double] = {
    val delta = 1e-4
    var grad = DenseMatrix.zeros[Double](x.rows, x.cols)

    // TODO: memory eff?
//    (0 until x.size).foreach { i =>
//      println(s"idx ${i}")
//      val xHigh = x.copy
//      xHigh.data(i) = xHigh.data(i) + delta
//      val xLow = x.copy
//      xLow.data(i) = xLow.data(i) - delta
//      grad.data(i) = numericalDiff(f, xHigh, xLow, delta)
//    }

    (0 until x.size).foreach { i =>
      Logger.trace(s"idx ${i}")
      val org = x.data(i)
      x.data(i) = org + delta
      val fHigh = f(x)
      x.data(i) = org - delta
      val fLow = f(x)
      grad.data(i) = (fHigh - fLow) / (2 * delta)
      Logger.trace(s"grad ${grad.data(i)}")
    }

    grad
  }

}

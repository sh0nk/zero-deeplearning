package com.github.sh0nk.zerodl.ch01

import breeze.linalg._

object MathCalculate {
  def main(args: Array[String]): Unit = {
    val x = DenseVector(1.0, 2.0, 3.0)
    val y = DenseVector(2.0, 4.0, 6.0)

    println(x + y)
  }
}

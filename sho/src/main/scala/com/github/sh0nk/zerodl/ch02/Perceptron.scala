package com.github.sh0nk.zerodl.ch02

abstract class Perceptron(w1: Double, w2: Double, theta: Double) {
  def OP(a: Double, b: Double): Int = {
    w1 * a + w2 * b + theta match {
      case x if x <= 0 => 0
      case _ => 1
    }
  }
}

class AND extends Perceptron(0.5, 0.5, -0.7) {}

class OR extends Perceptron(0.5, 0.5, -0.2) {}

class NAND extends Perceptron(-0.5, -0.5, 0.7) {}

class XOR extends Perceptron(0, 0, 0) {
  override def OP(a: Double, b: Double): Int = {
    (new AND).OP((new NAND).OP(a, b), (new OR).OP(a, b))
  }
}

object Test {
  def main(args: Array[String]): Unit = {
    val and = new AND
    println(and.OP(0, 0))
    println(and.OP(1, 0))
    println(and.OP(0, 1))
    println(and.OP(1, 1))

    val xor = new XOR
    println(xor.OP(0, 0))
    println(xor.OP(1, 0))
    println(xor.OP(0, 1))
    println(xor.OP(1, 1))
  }
}
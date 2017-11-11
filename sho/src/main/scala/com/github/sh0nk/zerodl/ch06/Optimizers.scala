package com.github.sh0nk.zerodl.ch06

import breeze.numerics.sqrt
import com.github.sh0nk.zerodl.ch04.Logger


case class SGD(learningRate: Double) extends Optimizer {

  override def perMatrixKey(key: String, layer: WeightLayer): Unit = {
    layer.W(key) -= layer.dW(key) * learningRate
  }

  override def perVectorKey(key: String, layer: WeightLayer): Unit = {
    layer.Wb(key) -= layer.dWb(key) * learningRate
  }
}

case class Momentum(learningRate: Double = .01, momentum: Double = .9) extends Optimizer {
  val momentumPrefix = "momentum_"

  override def perMatrixKey(key: String, layer: WeightLayer): Unit = {
    initInternalMatWeight(momentumPrefix + key, layer, layer.W(key).rows, layer.W(key).cols)
    layer.dW += (momentumPrefix + key -> (layer.dW(momentumPrefix + key) * momentum - layer.dW(key) * learningRate))
    Logger.trace(layer.dW(momentumPrefix + key)(0, ::))
    layer.W(key) += layer.dW(momentumPrefix + key)
  }

  override def perVectorKey(key: String, layer: WeightLayer): Unit = {
    initInternalVecWeight(momentumPrefix + key, layer, layer.Wb(key).length)
    layer.dWb += (momentumPrefix + key -> (layer.dWb(momentumPrefix + key) * momentum - layer.dWb(key) * learningRate))
    layer.Wb(key) += layer.dWb(momentumPrefix + key)
  }
}

case class AdaGrad(learningRate: Double = .01) extends Optimizer {
  val adaGradPrefix = "adagrad_"

  override def perMatrixKey(key: String, layer: WeightLayer): Unit = {
    initInternalMatWeight(adaGradPrefix + key, layer, layer.W(key).rows, layer.W(key).cols)
    layer.dW(adaGradPrefix + key) += layer.dW(key) *:* layer.dW(key)
    Logger.trace(layer.dW(adaGradPrefix + key)(0, ::))
    layer.W(key) -= layer.dW(key) * learningRate / (sqrt(layer.dW(adaGradPrefix + key)) + 1e-7)
  }

  override def perVectorKey(key: String, layer: WeightLayer): Unit = {
    initInternalVecWeight(adaGradPrefix + key, layer, layer.Wb(key).length)
    layer.dWb(adaGradPrefix + key) += layer.dWb(key) *:* layer.dWb(key)
    Logger.trace(layer.dWb(adaGradPrefix + key)(0))
    layer.Wb(key) -= layer.dWb(key) * learningRate / (sqrt(layer.dWb(adaGradPrefix + key)) + 1e-7)
  }
}
package com.github.sh0nk.zerodl.ch06

import breeze.linalg.{DenseMatrix, DenseVector}
import com.github.sh0nk.zerodl.ch04.Logger
import com.github.sh0nk.zerodl.ch05.Layer

/**
  * Expected only one per each layer, no multiple optimizer can be applied to a layer.
  */
trait Optimizer {
  final def optimize(layers: Seq[Layer]): Unit = {
    layers.filter(_.isInstanceOf[WeightLayer]).map(_.asInstanceOf[WeightLayer]).foreach { layer =>
      layer.W.keys.foreach { key =>
        Logger.trace(s"gradientDescent key for W: $key")
        perMatrixKey(key, layer)
      }
      layer.Wb.keys.foreach { key =>
        Logger.trace(s"gradientDescent key for b: $key")
        perVectorKey(key, layer)
      }
    }
  }

  protected def initInternalMatWeight(completedKey: String, layer: WeightLayer, rows: Int, cols: Int): Unit = {
    if (!layer.dW.contains(completedKey)) {
      Logger.info(s"init mat Weight for key ${completedKey}")
      layer.dW += (completedKey -> DenseMatrix.zeros[Double](rows, cols))
    }
  }

  protected def initInternalVecWeight(completedKey: String, layer: WeightLayer, length: Int): Unit = {
    if (!layer.dWb.contains(completedKey)) {
      Logger.info(s"init vec Weight for key ${completedKey}")
      layer.dWb += (completedKey -> DenseVector.zeros[Double](length))
    }
  }

  protected def perMatrixKey(key: String, layer: WeightLayer)

  protected def perVectorKey(key: String, layer: WeightLayer)
}

package com.github.sh0nk.zerodl.ch06

import breeze.linalg.{DenseMatrix, DenseVector}
import com.github.sh0nk.matplotlib4j.{NumpyUtils, Plot}
import com.github.sh0nk.zerodl.ch04.Logger

class OptimizersOnContour(optimizer: Optimizer) {
  val xyLayer = new WeightLayer {
    Wb = Map("xy" -> DenseVector(-7.0, 2.0)) // params
    dWb = Map("xy" -> DenseVector(0.0, 0.0)) // grad

    override def forward(x: DenseMatrix[Double], train: Boolean = false): DenseMatrix[Double] = ???
    override def backward(dout: DenseMatrix[Double]): DenseMatrix[Double] = ???
  }

  var xHistory: Seq[Double] = Seq()
  var yHistory: Seq[Double] = Seq()

  def df(xy: DenseVector[Double]): DenseVector[Double] = {
    DenseVector[Double](xy.valueAt(0) / 10.0, xy.valueAt(1) * 2.0)
  }

  def iter(): Unit = {
    xHistory :+= xyLayer.Wb("xy").valueAt(0)
    yHistory :+= xyLayer.Wb("xy").valueAt(1)

    xyLayer.dWb += ("xy" -> df(xyLayer.Wb("xy")))
    optimizer.optimize(Seq(xyLayer))

    Logger.info(s"current param: ${xyLayer.Wb("xy")}")
    Logger.info(s"current grad: ${xyLayer.dWb("xy")}")
  }

  def addPlot(plt: Plot): Unit = {
    Range(0, 30).foreach(_ -> iter)
    import collection.JavaConverters._
    plt.plot().add(xHistory.map(Double.box).toList.asJava, yHistory.map(Double.box).toList.asJava, "ro-")
  }

}

object OptimizersOnContour {
  def setContour(plt: Plot): Unit = {
    val x = NumpyUtils.arange(-10, 10, 0.01)
    val y = NumpyUtils.arange(-5, 5, 0.01)
    val z = NumpyUtils.meshgrid(x, y)
    println(z.x)
    println(z.y)
    val calcedZ = z.calcZ[java.lang.Double]((xi, yj) => {
      var v = xi * xi / 20 + yj * yj
      if (v > 7) v = 0
      v
    })
    plt.contour().add(x, y, calcedZ)
  }

  def draw(optimizer: Optimizer): Unit = {
    val plt = Plot.create()
    setContour(plt)
    new OptimizersOnContour(optimizer).addPlot(plt)
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)

    import collection.JavaConverters._
    plt.plot().add(List(Double.box(0)).asJava, List(Double.box(0)).asJava, "+")
    plt.show()
  }

  def main(args: Array[String]): Unit = {
    draw(SGD(0.95))
    draw(Momentum(0.1))
    draw(AdaGrad(1.5))
  }

}
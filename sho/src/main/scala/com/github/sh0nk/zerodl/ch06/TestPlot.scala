package com.github.sh0nk.zerodl.ch06

import com.github.sh0nk.matplotlib4j.{NumpyUtils, Plot}

object TestPlot {
  def main(args: Array[String]): Unit = {
    import scala.collection.JavaConverters._
    val x = NumpyUtils.linspace(-3, 3, 100).asScala.toList
    val y = x.map(xi => Math.sin(xi) + Math.random()).map(Double.box)

    val plt = Plot.create()
    plt.plot().add(x.asJava, y.asJava, "o")
    plt.title("scatter")
    plt.show()
  }

}

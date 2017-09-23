package com.github.sh0nk.zerodl.ch03

import java.io.{BufferedInputStream, _}
import java.util.zip.GZIPInputStream

class MNISTLoader {
  private def load2DimIntArray(fileName: String, dim2: Int = 784, header: Int = 16): Array[Array[Int]] = {
    val is = new DataInputStream(new BufferedInputStream(new GZIPInputStream(new FileInputStream(fileName))))
    is.skipBytes(header) // remove header

    val it = new Iterator[Array[Int]] {
      var cnt = 0
      var checked = false
      var buf: Array[Int] = null

      override def hasNext: Boolean = {
        checked = true
        try {
          buf = (0 until dim2).map(_ => is.readUnsignedByte()).toArray
          true
        } catch {
          case e: EOFException =>
            buf = null
            false
        }
      }

      override def next(): Array[Int] = {
        if (!checked) {
          hasNext
        }
        checked = false
        buf
      }
    }

    it.toArray
  }

  def loadImage(fileName: String, normalize: Boolean = true): Array[Array[Double]] = {
    val orgImg = load2DimIntArray(fileName, 784, 16)

    if (normalize) {
      orgImg.map(_.map(_.toDouble / 255.0))
    } else {
      orgImg.map(_.map(_.toDouble))
    }
  }

  def loadLabel(fileName: String, oneHot: Boolean = true): Array[Array[Int]] = {
    val orgLabel = load2DimIntArray(fileName, 1, 8)

    if (oneHot) {
      orgLabel.map { v =>
        val cls = v(0)
        (0 until 10).map {
          case `cls` => 1
          case _ => 0
        }.toArray
      }
    } else {
      orgLabel
    }
  }
}

object MNISTLoader {
  def showImg(img: Array[Double]) {
    img.zipWithIndex.foreach { case (v, i) =>
      if (v > 0) {
        print(s" 1")
      } else {
        print(s" 0")
      }
      if ((i + 1) % 28 == 0) println()
    }
  }

  def main(args: Array[String]): Unit = {
    val loader = new MNISTLoader()
    val img = loader.loadImage("/var/folders/6t/dlnd80jj05s8r8qcjmgm4q5cv9pfnn/T/zerodl-mnist/train-images-idx3-ubyte.gz", normalize = false)
    println(s"img num ${img.length}")
    println(s"img byte length ${img(0).length}")
    showImg(img(0))
    println()
    showImg(img(1))
    println()

    val lbl = loader.loadLabel("/var/folders/6t/dlnd80jj05s8r8qcjmgm4q5cv9pfnn/T/zerodl-mnist/train-labels-idx1-ubyte.gz", oneHot = true)
    println(s"label num ${lbl.length}")
    println(s"label byte length ${lbl(0).length}")
    lbl(0).foreach(v => print(s" ${v}"))
    println()
    lbl(1).foreach(v => print(s" ${v}"))
    println()
  }
}
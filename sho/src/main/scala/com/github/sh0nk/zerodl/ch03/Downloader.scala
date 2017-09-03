package com.github.sh0nk.zerodl.ch03

import java.io.{BufferedOutputStream, File, FileOutputStream}
import java.net.URL
import java.nio.file.Files

class Downloader {
  private def download(url: String, outputFile: String) = {
    println(s"start downloading URI: ${url} to file: ${outputFile}")
    val stream = new URL(url).openStream()
    val buffer = Stream.continually(stream.read()).takeWhile(-1 != ).map(_.byteValue).toArray
    val bw = new BufferedOutputStream(new FileOutputStream(outputFile))
    bw.write(buffer)
    bw.close()
    println(s"download finished URI: ${url} to file: ${outputFile}")
  }

  def ensureDownloadingAll() = {
    Downloader.keyFile.foreach {
      case (k, v) =>
        val file = new File(Downloader.baseDir.toFile, v).toPath
        if (!Files.exists(file)) {
          download(Downloader.urlBase + v, file.toString)
        } else {
          println(s"file $v exists")
        }
    }
  }
}

object Downloader {
  val urlBase = "http://yann.lecun.com/exdb/mnist/"
  val keyFile = Map(
    "train_img" -> "train-images-idx3-ubyte.gz",
    "train_label" -> "train-labels-idx1-ubyte.gz",
    "test_img" -> "t10k-images-idx3-ubyte.gz",
    "test_label" -> "t10k-labels-idx1-ubyte.gz"
  )
  val baseDir = {
    val dir = new File(System.getProperty("java.io.tmpdir"), "zerodl-mnist").toPath
    if (!Files.exists(dir)) {
      Files.createDirectory(dir)
      println(s"Temp directory created ${dir}")
    }
    dir.toAbsolutePath
  }

  def main(args: Array[String]) = {
    new Downloader().ensureDownloadingAll()
  }
}
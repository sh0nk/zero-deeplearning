package com.github.sh0nk.zerodl.ch04

object Logger {
  val logLevel = LogLevels.INFO

  object LogLevels extends Enumeration {
    type LogLevel = Value
    val ERROR = Value(0)
    val WARN = Value(1)
    val INFO = Value(2)
    val DEBUG = Value(3)
    val TRACE = Value(4)
  }

  def error[T](a: T) = if (logLevel.id >= LogLevels.ERROR.id) println(a)
  def warn[T](a: T) = if (logLevel.id >= LogLevels.WARN.id) println(a)
  def info[T](a: T) = if (logLevel.id >= LogLevels.INFO.id) println(a)
  def debug[T](a: T) = if (logLevel.id >= LogLevels.DEBUG.id) println(a)
  def trace[T](a: T) = if (logLevel.id >= LogLevels.TRACE.id) println(a)

}
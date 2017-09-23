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

  def error[T](a: T) = println2(a, LogLevels.ERROR)
  def warn[T](a: T) = println2(a, LogLevels.WARN)
  def info[T](a: T) = println2(a, LogLevels.INFO)
  def debug[T](a: T) = println2(a, LogLevels.DEBUG)
  def trace[T](a: T) = println2(a, LogLevels.TRACE)

  private def println2[T](a: T, logLevels: Logger.LogLevels.Value) =
    if (logLevel.id >= logLevels.id) println(s"$logLevels: $a")
}
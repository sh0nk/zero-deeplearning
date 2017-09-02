/**
 * @author Noriyuki Ishida
 */
package io.github.noriyuki106.numkt

import java.util.Arrays

class NumericArray(private val values: DoubleArray) {
    val length = this.values.size

    override fun toString(): String {
        return Arrays.toString(this.values)
    }

    operator fun get(i: Int): Double = this.values[i]

    operator fun plus(that: NumericArray): NumericArray {
        if (this.length != that.length) throw IllegalArgumentException()
        return narrayOf(*this.values.mapIndexed { idx, v -> v + that.values[idx] }.toDoubleArray())
    }

    operator fun plus(that: Double): NumericArray = narrayOf(
            *this.values.map { it + that }.toDoubleArray())

    operator fun plus(that: Int): NumericArray = narrayOf(
            *this.values.map { it + that }.toDoubleArray())

    operator fun minus(that: NumericArray): NumericArray {
        if (this.length != that.length) throw IllegalArgumentException()
        return narrayOf(*this.values.mapIndexed { idx, v -> v - that.values[idx] }.toDoubleArray())
    }

    operator fun minus(that: Double): NumericArray = narrayOf(
            *this.values.map { it - that }.toDoubleArray())

    operator fun minus(that: Int): NumericArray = narrayOf(
            *this.values.map { it - that }.toDoubleArray())

    operator fun times(that: NumericArray): Double {
        if (this.length != that.length) throw IllegalArgumentException()
        return this.values.mapIndexed { idx, v -> v * that.values[idx] }.sum()
    }

    operator fun times(that: Double): NumericArray = narrayOf(
            *this.values.map { it * that }.toDoubleArray())

    operator fun times(that: Int): NumericArray = narrayOf(
            *this.values.map { it * that }.toDoubleArray())

    operator fun div(that: NumericArray): NumericArray {
        if (this.length != that.length) throw IllegalArgumentException()
        return narrayOf(*this.values.mapIndexed { idx, v -> v / that.values[idx] }.toDoubleArray())
    }

    operator fun div(that: Double): NumericArray = narrayOf(
            *this.values.map { it / that }.toDoubleArray())

    operator fun div(that: Int): NumericArray = narrayOf(
            *this.values.map { it / that }.toDoubleArray())

    fun map(transformer: (Double) -> Double): NumericArray = narrayOf(
            *this.values.map { transformer(it) }.toDoubleArray())

    fun mapIndexed(transformer: (Int, Double) -> Double): NumericArray = narrayOf(
            *this.values.mapIndexed { index, value -> transformer(index, value) }.toDoubleArray())

    fun sum(): Double = this.values.sum()
    fun sumBy(transformer: (Double) -> Double): Double = this.values.sumByDouble { transformer(it) }

    fun max(): Double = this.values.max() ?: 0.0

    fun removed(i: Int): NumericArray = narrayOf(*this.values.copyOfRange(0, i),
            *this.values.copyOfRange(i + 1, this.length))

    fun inserted(v: Double, i: Int): NumericArray = narrayOf(*this.values.copyOfRange(0, i),
            v, *this.values.copyOfRange(i, this.length))
}

fun narrayOf(vararg values: Double) = NumericArray(values)
fun narrayOf(vararg values: Int) = NumericArray(values.map { it.toDouble() }.toDoubleArray())

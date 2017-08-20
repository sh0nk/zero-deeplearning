/**
 * @author Noriyuki Ishida
 */
package io.github.noriyuki106.numkt

import java.util.Arrays

class NumericArray(val values: DoubleArray) {
    val length = this.values.size

    override fun toString(): String {
        return this.values.toString()
    }

    operator fun get(i: Int): Double = this.values[i]
}

fun narrayOf(vararg values: Double) = NumericArray(values)

operator fun NumericArray.plus(that: NumericArray): NumericArray {
    if (this.length != that.length) throw IllegalArgumentException()
    return narrayOf(*this.values.mapIndexed { idx, v -> v + that.values[idx] }.toDoubleArray())
}
operator fun NumericArray.plus(that: Double): NumericArray = narrayOf(*this.values.map { it + that }.toDoubleArray())
operator fun NumericArray.plus(that: Int): NumericArray = narrayOf(*this.values.map { it + that }.toDoubleArray())

operator fun NumericArray.minus(that: NumericArray): NumericArray {
    if (this.length != that.length) throw IllegalArgumentException()
    return narrayOf(*this.values.mapIndexed { idx, v -> v - that.values[idx] }.toDoubleArray())
}

operator fun NumericArray.minus(that: Double): NumericArray = narrayOf(*this.values.map { it - that }.toDoubleArray())
operator fun NumericArray.minus(that: Int): NumericArray = narrayOf(*this.values.map { it - that }.toDoubleArray())

operator fun NumericArray.times(that: NumericArray): NumericArray {
    if (this.length != that.length) throw IllegalArgumentException()
    return narrayOf(*this.values.mapIndexed { idx, v -> v * that.values[idx] }.toDoubleArray())
}

operator fun NumericArray.times(that: Double): NumericArray = narrayOf(*this.values.map { it * that }.toDoubleArray())
operator fun NumericArray.times(that: Int): NumericArray = narrayOf(*this.values.map { it * that }.toDoubleArray())

operator fun NumericArray.div(that: NumericArray): NumericArray {
    if (this.length != that.length) throw IllegalArgumentException()
    return narrayOf(*this.values.mapIndexed { idx, v -> v / that.values[idx] }.toDoubleArray())
}

operator fun NumericArray.div(that: Double): NumericArray = narrayOf(*this.values.map { it / that }.toDoubleArray())
operator fun NumericArray.div(that: Int): NumericArray = narrayOf(*this.values.map { it / that }.toDoubleArray())

fun NumericArray.sum(): Double = this.values.sum()

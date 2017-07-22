/**
 * @author Noriyuki Ishida
 */
package io.github.noriyuki106.numkt

import java.util.Arrays

class NumericArray<out T : Number>(val values: Array<out T>) {
    val length = this.values.size

    override fun toString(): String {
        return Arrays.deepToString(this.values)
    }

    operator fun get(i: Int): T = this.values[i]
}

fun <T : Number> narrayOf(vararg values: T) = NumericArray(values)

operator fun NumericArray<Double>.plus(that: NumericArray<Double>): NumericArray<Double> {
    if (this.length != that.length) throw IllegalArgumentException()
    return narrayOf(*this.values.mapIndexed { idx, v -> v + that.values[idx] }.toTypedArray())
}
operator fun NumericArray<Double>.plus(that: Double): NumericArray<Double> = narrayOf(*this.values.map { it + that }.toTypedArray())
operator fun NumericArray<Double>.plus(that: Int): NumericArray<Double> = narrayOf(*this.values.map { it + that }.toTypedArray())

operator fun NumericArray<Double>.minus(that: NumericArray<Double>): NumericArray<Double> {
    if (this.length != that.length) throw IllegalArgumentException()
    return narrayOf(*this.values.mapIndexed { idx, v -> v - that.values[idx] }.toTypedArray())
}

operator fun NumericArray<Double>.minus(that: Double): NumericArray<Double> = narrayOf(*this.values.map { it - that }.toTypedArray())
operator fun NumericArray<Double>.minus(that: Int): NumericArray<Double> = narrayOf(*this.values.map { it - that }.toTypedArray())

operator fun NumericArray<Double>.times(that: NumericArray<Double>): NumericArray<Double> {
    if (this.length != that.length) throw IllegalArgumentException()
    return narrayOf(*this.values.mapIndexed { idx, v -> v * that.values[idx] }.toTypedArray())
}

operator fun NumericArray<Double>.times(that: Double): NumericArray<Double> = narrayOf(*this.values.map { it * that }.toTypedArray())
operator fun NumericArray<Double>.times(that: Int): NumericArray<Double> = narrayOf(*this.values.map { it * that }.toTypedArray())

operator fun NumericArray<Double>.div(that: NumericArray<Double>): NumericArray<Double> {
    if (this.length != that.length) throw IllegalArgumentException()
    return narrayOf(*this.values.mapIndexed { idx, v -> v / that.values[idx] }.toTypedArray())
}

operator fun NumericArray<Double>.div(that: Double): NumericArray<Double> = narrayOf(*this.values.map { it / that }.toTypedArray())
operator fun NumericArray<Double>.div(that: Int): NumericArray<Double> = narrayOf(*this.values.map { it / that }.toTypedArray())

fun NumericArray<Double>.sum(): Double = this.values.sum()

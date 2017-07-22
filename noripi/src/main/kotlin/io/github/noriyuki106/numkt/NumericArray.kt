/**
 * Copyright (C) 2017 Retty, Inc.
 *
 * This program is free software: you can redistribute it and/or modify it under the terms of the
 * GNU General Public License as published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
 * even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with this program. If
 * not, see <http://www.gnu.org/licenses/>.
 *
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

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

class Matrix<T : Number>(val values: Array<out NumericArray<T>>) {
    val rowSize = this.values.size
    val colSize = this.values.firstOrNull()?.length ?: 0

    override fun toString(): String {
        return Arrays.deepToString(this.values)
    }

    operator fun get(i: Int, j: Int): T = this.values[i][j]
}

fun <T : Number> matrixOf(vararg values: NumericArray<T>) = Matrix(values)

operator fun Matrix<Double>.plus(that: Matrix<Double>): Matrix<Double> {
    if (this.rowSize != that.rowSize || this.colSize != that.colSize) throw IllegalArgumentException()

    return matrixOf(*this.values.mapIndexed { idx, arr -> arr + that.values[idx] }.toTypedArray())
}

operator fun Matrix<Double>.plus(that: Double): Matrix<Double> = matrixOf(*this.values.map { it + that }.toTypedArray())
operator fun Matrix<Double>.plus(that: Int): Matrix<Double> = matrixOf(*this.values.map { it + that }.toTypedArray())

operator fun Matrix<Double>.minus(that: Matrix<Double>): Matrix<Double> {
    if (this.rowSize != that.rowSize || this.colSize != that.colSize) throw IllegalArgumentException()

    return matrixOf(*this.values.mapIndexed { idx, arr -> arr - that.values[idx] }.toTypedArray())
}

operator fun Matrix<Double>.minus(that: Double): Matrix<Double> = matrixOf(*this.values.map { it - that }.toTypedArray())
operator fun Matrix<Double>.minus(that: Int): Matrix<Double> = matrixOf(*this.values.map { it - that }.toTypedArray())

operator fun Matrix<Double>.times(that: Double): Matrix<Double> = matrixOf(*this.values.map { it * that }.toTypedArray())
operator fun Matrix<Double>.times(that: Int): Matrix<Double> = matrixOf(*this.values.map { it * that }.toTypedArray())

operator fun Matrix<Double>.div(that: Double): Matrix<Double> = matrixOf(*this.values.map { it / that }.toTypedArray())
operator fun Matrix<Double>.div(that: Int): Matrix<Double> = matrixOf(*this.values.map { it / that }.toTypedArray())

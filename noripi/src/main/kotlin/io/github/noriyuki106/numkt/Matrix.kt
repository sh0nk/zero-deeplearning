/**
 * @author Noriyuki Ishida
 */
package io.github.noriyuki106.numkt

import java.util.Arrays

class Matrix(val values: Array<out NumericArray>) {
    val rowSize = this.values.size
    val colSize = this.values.firstOrNull()?.length ?: 0

    val shape = "(${this.rowSize}, ${this.colSize})"

    override fun toString(): String {
        return Arrays.deepToString(this.values)
    }

    operator fun get(i: Int, j: Int): Double = this.values[i][j]

    fun getRow(i: Int): NumericArray {
        return this.values[i]
    }

    fun getCol(j: Int): NumericArray {
        return NumericArray(this.values.map { it[j] }.toDoubleArray())
    }
}

fun matrixOf(vararg values: NumericArray) = Matrix(values)

operator fun Matrix.plus(that: Matrix): Matrix {
    if (this.rowSize != that.rowSize || this.colSize != that.colSize) throw IllegalArgumentException()

    return matrixOf(*this.values.mapIndexed { idx, arr -> arr + that.values[idx] }.toTypedArray())
}

operator fun Matrix.plus(that: Double): Matrix = matrixOf(*this.values.map { it + that }.toTypedArray())
operator fun Matrix.plus(that: Int): Matrix = matrixOf(*this.values.map { it + that }.toTypedArray())

operator fun Matrix.minus(that: Matrix): Matrix {
    if (this.rowSize != that.rowSize || this.colSize != that.colSize) throw IllegalArgumentException()

    return matrixOf(*this.values.mapIndexed { idx, arr -> arr - that.values[idx] }.toTypedArray())
}

operator fun Matrix.minus(that: Double): Matrix = matrixOf(*this.values.map { it - that }.toTypedArray())
operator fun Matrix.minus(that: Int): Matrix = matrixOf(*this.values.map { it - that }.toTypedArray())

operator fun Matrix.times(that: Double): Matrix = matrixOf(*this.values.map { it * that }.toTypedArray())
operator fun Matrix.times(that: Int): Matrix = matrixOf(*this.values.map { it * that }.toTypedArray())

operator fun Matrix.div(that: Double): Matrix = matrixOf(*this.values.map { it / that }.toTypedArray())
operator fun Matrix.div(that: Int): Matrix = matrixOf(*this.values.map { it / that }.toTypedArray())


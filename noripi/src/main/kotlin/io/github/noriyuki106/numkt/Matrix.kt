/**
 * @author Noriyuki Ishida
 */
package io.github.noriyuki106.numkt

import java.util.Arrays

class Matrix<out T : Number>(val values: Array<out NumericArray<T>>) {
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

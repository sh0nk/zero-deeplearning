/**
 * @author Noriyuki Ishida
 */
package io.github.noriyuki106.numkt

import kotlinx.coroutines.experimental.yield
import java.util.Arrays
import kotlin.coroutines.experimental.buildIterator

class Matrix(val values: Array<out NumericArray>) {
    val rowSize = this.values.size
    val colSize = this.values.firstOrNull()?.length ?: 0

    val shape = "(${this.rowSize}, ${this.colSize})"

    override fun toString(): String {
        return Arrays.deepToString(this.values)
    }

    operator fun get(i: Int, j: Int): Double = this.values[i][j]

    fun rowIterator(): Iterator<NumericArray> = buildIterator {
        (0 until this@Matrix.rowSize).forEach {
            yield(this@Matrix.values[it])
        }
    }

    fun colIterator(): Iterator<NumericArray> = buildIterator {
        (0 until this@Matrix.colSize).forEach { idx ->
            yield(narrayOf(*this@Matrix.values.map { it[idx] }.toDoubleArray()))
        }
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

operator fun Matrix.times(that: Matrix): Matrix {
    if (this.colSize != that.rowSize) throw IllegalArgumentException()

    var rows = mutableListOf<NumericArray>()
    this.rowIterator().forEach { row ->
        var newRow = mutableListOf<Double>()
        that.colIterator().forEach { col ->
            newRow.add(row * col)
        }

        rows.add(narrayOf(*newRow.toDoubleArray()))
    }

    return matrixOf(*rows.toTypedArray())
}

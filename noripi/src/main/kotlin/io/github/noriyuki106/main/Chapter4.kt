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
package io.github.noriyuki106.main

import io.github.noriyuki106.neural_network.crossEntropyError
import io.github.noriyuki106.neural_network.meanSquaredError
import io.github.noriyuki106.numkt.NumericArray
import io.github.noriyuki106.numkt.function.curryByFirst
import io.github.noriyuki106.numkt.function.curryBySecond
import io.github.noriyuki106.numkt.function.draw
import io.github.noriyuki106.numkt.function.numericalDiff
import io.github.noriyuki106.numkt.function.numericalGradient
import io.github.noriyuki106.numkt.narrayOf

fun main(args: Array<String>) {
//    sample4_2_1()
//    sample4_2_2()
//    sample4_3_2()
//    sample4_3_3()
    sample4_4()
    sample4_4_1()
}

private fun sample4_2_1() {
    val t = narrayOf(0, 0, 1, 0, 0, 0, 0, 0, 0, 0)

    val y1 = narrayOf(0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0)
    println(meanSquaredError(y1, t))

    val y2 = narrayOf(0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0)
    println(meanSquaredError(y2, t))
}

private fun sample4_2_2() {
    val t = narrayOf(0, 0, 1, 0, 0, 0, 0, 0, 0, 0)
    val y1 = narrayOf(0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0)
    println(crossEntropyError(t, y1))

    val y2 = narrayOf(0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0)
    println(crossEntropyError(t, y2))
}

private fun sample4_3_2() {
    val function = fun(x: Double): Double = 0.01 * x * x + 0.1 * x

    function.draw(0.0..20.0)
    println(function.numericalDiff(5.0))
    println(function.numericalDiff(10.0))
}

private fun sample4_3_3() {
    val function = fun(x1: Double, x2: Double): Double = x1 * x1 + x2 * x2

    println(function.curryBySecond()(4.0).numericalDiff(3.0))
    println(function.curryByFirst()(3.0).numericalDiff(4.0))
}

private fun sample4_4() {
    val function = fun(x1: Double, x2: Double): Double = x1 * x1 + x2 * x2

    println(function.numericalGradient(narrayOf(3.0, 4.0)))
    println(function.numericalGradient(narrayOf(0.0, 2.0)))
    println(function.numericalGradient(narrayOf(3.0, 0.0)))
}

private fun sample4_4_1() {
    val function = fun(x: NumericArray): Double = x * x

    println(function.numericalGradient(narrayOf(3.0, 4.0)))
    println(function.numericalGradient(narrayOf(0.0, 2.0)))
    println(function.numericalGradient(narrayOf(3.0, 0.0)))

//    println(function.applyGradientMethod()
}
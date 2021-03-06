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
package io.github.noriyuki106.numkt.function

import io.github.noriyuki106.numkt.NumericArray
import io.github.noriyuki106.numkt.narrayOf

typealias NumericBiFunction = (Double, Double) -> Double

fun NumericBiFunction.curryByFirst(): (Double) -> NumericFunction {
    return fun (x1: Double): NumericFunction {
        return fun (x2: Double): Double {
            return this@curryByFirst(x1, x2)
        }
    }
}

fun NumericBiFunction.curryBySecond(): (Double) -> NumericFunction {
    return fun (x2: Double): NumericFunction {
        return fun (x1: Double): Double {
            return this@curryBySecond(x1, x2)
        }
    }
}

fun NumericBiFunction.numericalGradient(x: NumericArray): NumericArray {
    return narrayOf(
            this.curryBySecond()(x[1]).numericalDiff(x[0]),
            this.curryBySecond()(x[0]).numericalDiff(x[1])
    )
}

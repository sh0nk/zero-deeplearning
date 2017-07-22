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
package io.github.noriyuki106

import io.github.noriyuki106.numkt.narrayOf
import io.github.noriyuki106.numkt.sum
import io.github.noriyuki106.numkt.times

typealias TwoDimPerceptron = (Double, Double) -> Double

fun twoDimPerceptron(bias: Double, weight1: Double, weight2: Double): TwoDimPerceptron {
    return fun (x1: Double, x2: Double): Double {
        val weights = narrayOf(weight1, weight2)
        val values = narrayOf(x1, x2)
        val result = bias + (weights * values).sum()
        return if (result <= 0) 0.0 else 1.0
    }
}

fun compositePerceptron(bias: Double, weight1: Double, weight2: Double): (TwoDimPerceptron, TwoDimPerceptron) -> TwoDimPerceptron {
    return fun (p1: TwoDimPerceptron, p2: TwoDimPerceptron): TwoDimPerceptron {
        return fun (x1: Double, x2: Double): Double {
            val weights = narrayOf(weight1, weight2)
            val values = narrayOf(p1(x1, x2), p2(x1, x2))
            val result = bias + (weights * values).sum()

            return if (result <= 0) 0.0 else 1.0
        }
    }
}

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
package io.github.noriyuki106.chapter2

import io.github.noriyuki106.numkt.narrayOf
import io.github.noriyuki106.numkt.sum
import io.github.noriyuki106.numkt.times

class Perceptron(
        private val bias: Double,
        private val weight1: Double,
        private val weight2: Double,
        private val p1: Perceptron? = null,
        private val p2: Perceptron? = null) {

    operator fun invoke(x1: Double, x2: Double): Double {
        val weights = narrayOf(this.weight1, this.weight2)
        val values = narrayOf(this.p1?.let { it(x1, x2) } ?: x1, this.p2?.let { it(x1, x2) } ?: x2)
        val result = this.bias + (weights * values).sum()
        return if (result <= 0) 0.0 else 1.0
    }

    operator fun invoke(p1: Perceptron, p2: Perceptron): Perceptron {
        return Perceptron(
                bias = this.bias,
                weight1 = this.weight1,
                weight2 = this.weight2,
                p1 = p1,
                p2 = p2
        )
    }
}

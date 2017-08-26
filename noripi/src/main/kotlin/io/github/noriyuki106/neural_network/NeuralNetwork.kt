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
package io.github.noriyuki106.neural_network

import io.github.noriyuki106.numkt.Matrix
import io.github.noriyuki106.numkt.NumericArray
import io.github.noriyuki106.numkt.matrixOf
import io.github.noriyuki106.numkt.plus
import io.github.noriyuki106.numkt.times

class NeuralNetwork(vararg private val layers: NeuralNetworkLayer) {
    operator fun invoke(input: NumericArray): NumericArray {
        return this.layers.fold(input) { acc, func -> func(acc) }
    }
}

data class NeuralNetworkLayer(val weight: Matrix,
                              val bias: NumericArray,
                              val activationFunction: ActivationFunction = sigmoid.toActivationFunction()) {
    operator fun invoke(input: NumericArray): NumericArray {
        // a1 = 1 * b1 + x1 * w11 + x2 * w21
        // a2 = 1 * b2 + x1 * w12 + x2 * w22
        // a3 = 1 * b3 + x1 * w13 + x2 * x23
        val beforeActivation = (matrixOf(input) * this.weight).rowIterator().next() + this.bias

        return beforeActivation.activateBy(this.activationFunction)
    }
}

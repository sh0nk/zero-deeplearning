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
package io.github.noriyuki106.chapter3

import io.github.noriyuki106.extension.draw
import io.github.noriyuki106.extension.truncate
import io.github.noriyuki106.neural_network.NeuralNetwork
import io.github.noriyuki106.neural_network.NeuralNetworkLayer
import io.github.noriyuki106.neural_network.identity
import io.github.noriyuki106.neural_network.relu
import io.github.noriyuki106.neural_network.sigmoid
import io.github.noriyuki106.neural_network.step
import io.github.noriyuki106.numkt.matrixOf
import io.github.noriyuki106.numkt.narrayOf
import io.github.noriyuki106.numkt.times

fun main(args: Array<String>) {
//    sample3_2()
//    sample3_3()
    sample3_4()
}

private fun sample3_2() {
    step.draw(-5.0..5.0)
    sigmoid.draw(-5.0..5.0)
    relu.draw(-5.0..5.0)
}

private fun sample3_3() {
    val A1 = matrixOf(narrayOf(1, 2), narrayOf(3, 4))
    val B1 = matrixOf(narrayOf(5, 6), narrayOf(7, 8))
    println(A1.shape) // (2, 2)
    println(B1.shape) // (2, 2)
    println(A1 * B1)

    val A2 = matrixOf(narrayOf(1, 2, 3), narrayOf(4, 5, 6))
    val B2 = matrixOf(narrayOf(1, 2), narrayOf(3, 4), narrayOf(5, 6))
    println(A2.shape) // (2, 3)
    println(B2.shape) // (3, 2)
    println(A2 * B2)
}

private fun sample3_4() {
    val network = NeuralNetwork(
            NeuralNetworkLayer(
                    weight = matrixOf(narrayOf(0.1, 0.3, 0.5), narrayOf(0.2, 0.4, 0.6)),
                    bias = narrayOf(0.1, 0.2, 0.3)
            ),
            NeuralNetworkLayer(
                    weight = matrixOf(narrayOf(0.1, 0.4), narrayOf(0.2, 0.5), narrayOf(0.3, 0.6)),
                    bias = narrayOf(0.1, 0.2)
            ),
            NeuralNetworkLayer(
                    weight = matrixOf(narrayOf(0.1, 0.3), narrayOf(0.2, 0.4)),
                    bias = narrayOf(0.1, 0.2),
                    activationFunction = identity
            )
    )

    println(network(narrayOf(1.0, 0.5)))
}


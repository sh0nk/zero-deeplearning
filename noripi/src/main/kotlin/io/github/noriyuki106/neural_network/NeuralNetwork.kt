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
import io.github.noriyuki106.numkt.function.NumericMatrixFunction
import io.github.noriyuki106.numkt.function.getMinimumValueByGradientMethod
import io.github.noriyuki106.numkt.matrixOf
import io.github.noriyuki106.numkt.narrayOf
import io.github.noriyuki106.numkt.times

class NeuralNetwork(vararg private val layers: NeuralNetworkLayer) {
    operator fun invoke(input: NumericArray, trueLabel: NumericArray): NumericArray {
        return this.layers.fold(input) { acc, layer ->
            layer.train(acc, trueLabel)
            return@fold layer(acc)
        }
    }
}

class NeuralNetworkLayer(private val bias: NumericArray,
                         internal val inputSize: Int,
                         internal val outputSize: Int,
                         private val activationFunction: ActivationFunction = sigmoid.toActivationFunction()) {

    private val lossEvaluator: (NumericArray, NumericArray) -> NumericMatrixFunction = { input, trueLabel ->
        { crossEntropyError(this(input, it), trueLabel) }
    }

    private var weight = Matrix.gaussianRandom(this.inputSize, this.outputSize)

    constructor(bias: Double = 0.0,
                inputSize: Int,
                outputSize: Int,
                activationFunction: ActivationFunction = sigmoid.toActivationFunction()) :

            this(
                    bias = narrayOf(*(0 until outputSize).map { bias }.toDoubleArray()),
                    inputSize = inputSize,
                    outputSize = outputSize,
                    activationFunction = activationFunction
            )

    fun train(input: NumericArray, trueLabel: NumericArray) {
        // new
        val newWeight = this.lossEvaluator(input, trueLabel).getMinimumValueByGradientMethod(
                initialValue = this.weight)

        println("loss: " + crossEntropyError(this(input), trueLabel))
        println("loss updated: " + crossEntropyError(this(input, newWeight), trueLabel))

        this.weight = newWeight
    }

    operator fun invoke(input: NumericArray, weight: Matrix = this.weight): NumericArray {
        // a1 = 1 * b1 + x1 * w11 + x2 * w21
        // a2 = 1 * b2 + x1 * w12 + x2 * w22
        // a3 = 1 * b3 + x1 * w13 + x2 * x23
        val beforeActivation = (matrixOf(input) * weight).rowIterator().next() + this.bias

        return beforeActivation.activateBy(this.activationFunction)
    }
}

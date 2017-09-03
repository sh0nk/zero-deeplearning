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

import io.github.noriyuki106.numkt.Either
import io.github.noriyuki106.numkt.Matrix
import io.github.noriyuki106.numkt.NumericArray
import io.github.noriyuki106.numkt.function.NumericMatrixFunction
import io.github.noriyuki106.numkt.function.getMinimumValueByGradientMethod
import io.github.noriyuki106.numkt.matrixOf
import io.github.noriyuki106.numkt.plus
import io.github.noriyuki106.numkt.times

class NeuralNetwork(vararg private val layers: NeuralNetworkLayer) {
    private val lossEvaluator: (NumericArray, NumericArray) -> NumericMatrixFunction = { input, trueLabel ->
        { crossEntropyError(this.predict(input, it), trueLabel) }
    }

    operator fun invoke(input: NumericArray, trueLabel: NumericArray): NumericArray {
        val weights = this.layers.map {
            Matrix.gaussianRandom(it.inputSize, it.outputSize)
        }

        println(weights.first())
        val matrixDiff = this.lossEvaluator(input, trueLabel).getMinimumValueByGradientMethod(
                initialValue = weights.first()
        )

        println(this.predict(input, weights.first()))

        return this.predict(input, weights.first() + matrixDiff)
    }

    private fun predict(input: NumericArray, weight: Matrix): NumericArray {
        return this.layers.foldIndexed(input) { _, acc, func ->
            func(acc, weight)
        }
    }
}

class NeuralNetworkLayer {
    private val activationFunction: ActivationFunction
    private val bias: Either<NumericArray, Double>
    internal val inputSize: Int
    internal val outputSize: Int

    constructor(bias: NumericArray,
                inputSize: Int,
                outputSize: Int,
                activationFunction: ActivationFunction = sigmoid.toActivationFunction()) {

        this.activationFunction = activationFunction
        this.inputSize = inputSize
        this.outputSize = outputSize
        this.bias = Either.left(bias)
    }

    constructor(bias: Double = 0.0,
                inputSize: Int,
                outputSize: Int,
                activationFunction: ActivationFunction = sigmoid.toActivationFunction()) {

        this.bias = Either.right(bias)
        this.inputSize = inputSize
        this.outputSize = outputSize
        this.activationFunction = activationFunction
    }

    operator fun invoke(input: NumericArray, weight: Matrix): NumericArray {
        // a1 = 1 * b1 + x1 * w11 + x2 * w21
        // a2 = 1 * b2 + x1 * w12 + x2 * w22
        // a3 = 1 * b3 + x1 * w13 + x2 * x23
        val beforeActivation = this.bias.reduce(
                leftTransformer = { arr: NumericArray ->
                    (matrixOf(input) * weight).rowIterator().next() + arr
                },
                rightTransformer = { num: Double ->
                    (matrixOf(input) * weight).rowIterator().next() + num
                }
        )

        return beforeActivation.activateBy(this.activationFunction)
    }
}

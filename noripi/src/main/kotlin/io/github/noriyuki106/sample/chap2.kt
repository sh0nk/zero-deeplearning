/**
 * @author Noriyuki Ishida
 */
package io.github.noriyuki106.sample

import io.github.noriyuki106.Perceptron


val and = Perceptron(bias = -0.7, weight1 = 0.5, weight2 = 0.5)
val nand = Perceptron(bias = 0.7, weight1 = -0.5, weight2 = -0.5)
val or = Perceptron(bias = -0.2, weight1 = 0.5, weight2 = 0.5)

val xor = and(nand, or)

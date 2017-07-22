/**
 * @author Noriyuki Ishida
 */
package io.github.noriyuki106.sample

import io.github.noriyuki106.compositePerceptron
import io.github.noriyuki106.twoDimPerceptron

val and = twoDimPerceptron(bias = -0.7, weight1 = 0.5, weight2 = 0.5)
val nand = twoDimPerceptron(bias = 0.7, weight1 = -0.5, weight2 = -0.5)
val or = twoDimPerceptron(bias = -0.2, weight1 = 0.5, weight2 = 0.5)

val xor = compositePerceptron(bias = -0.7, weight1 = 0.5, weight2 = 0.5)(nand, or)

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
package io.github.noriyuki106.sample

import io.github.noriyuki106.compositePerceptron
import io.github.noriyuki106.twoDimPerceptron

val and = twoDimPerceptron(bias = -0.7, weight1 = 0.5, weight2 = 0.5)
val nand = twoDimPerceptron(bias = 0.7, weight1 = -0.5, weight2 = -0.5)
val or = twoDimPerceptron(bias = -0.2, weight1 = 0.5, weight2 = 0.5)

val xor = compositePerceptron(bias = -0.7, weight1 = 0.5, weight2 = 0.5)(nand, or)

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

import io.github.noriyuki106.numkt.NumericArray
import io.github.noriyuki106.numkt.narrayOf

typealias ActivationFunction = (NumericArray) -> NumericArray

val sigmoid = fun (x: Double): Double = 1.0 / (1.0 + Math.exp(-x))
val step = fun (x: Double): Double = if (x > 0) 1.0 else 0.0
val relu = fun (x: Double): Double = Math.max(0.0, x)
val identity = fun (x: Double): Double = x
val softmax = fun (n: NumericArray): NumericArray = n.map { Math.exp(it - n.max()) / n.sumBy { Math.exp(it - n.max()) }}

fun NumericArray.activateBy(f: ActivationFunction) = f(this)
fun ((Double) -> Double).toActivationFunction() = fun (n: NumericArray): NumericArray = n.map { this(it) }

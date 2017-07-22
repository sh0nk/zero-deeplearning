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

import io.github.noriyuki106.numkt.div
import io.github.noriyuki106.numkt.matrixOf
import io.github.noriyuki106.numkt.minus
import io.github.noriyuki106.numkt.narrayOf
import io.github.noriyuki106.numkt.plus
import io.github.noriyuki106.numkt.times

fun sample1_5_3() {
    val x = narrayOf(1.0, 2.0, 3.0)
    val y = narrayOf(2.0, 4.0, 6.0)

    println(x + y)
    println(x - y)
    println(x * y)
    println(x / y)
}

fun sample1_5_4() {
    val x = matrixOf(narrayOf(1.0, 2.0), narrayOf(3.0, 4.0))
    val y = matrixOf(narrayOf(2.0, 4.0), narrayOf(6.0, 8.0))

    println(x[1, 1])
    println(x + y)
    println(x - y)
}

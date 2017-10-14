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
package io.github.noriyuki106.main

import io.github.noriyuki106.call_graph.CallGraph
import io.github.noriyuki106.call_graph.layer.AddLayer
import io.github.noriyuki106.call_graph.layer.MultiplyLayer
import io.github.noriyuki106.numkt.narrayOf

fun main(args: Array<String>) {
    sample5_4_1()
    sample5_4_2()
}

private fun sample5_4_1() {
    val apple = 100.0
    val appleNum = 2.0
    val tax = 1.1

    val callGraph = CallGraph(MultiplyLayer(), MultiplyLayer())

    val result = callGraph
            .forward { narrayOf(apple, appleNum) }
            .then { narrayOf(it[0], tax) }
            .done()

    val backwardResults = callGraph
            .backward { 1.0 }
            .then { it[0][0] }
            .done()

    println(result)
    println(backwardResults)
}

private fun sample5_4_2() {
    val apple = 100.0
    val appleNum = 2.0
    val mikan = 150.0
    val mikanNum = 3.0
    val tax = 1.1

    val callGraph = CallGraph(
            MultiplyLayer(), // calc apple subtotal
            MultiplyLayer(), // calc mikan subtotal
            AddLayer(), // calc subtotal (excl. tax)
            MultiplyLayer()
    )

    val result = callGraph
            .forward { narrayOf(apple, appleNum) }
            .then { narrayOf(mikan, mikanNum) }
            .then { narrayOf(it[0], it[1]) }
            .then { narrayOf(it[2], tax) }
            .done()

    val backwardResult = callGraph
            .backward { 1.0 }
            .then { it[0][0] }
            .then { it[1][1] }
            .then { it[1][0] }
            .done()

    println(result)
    println(backwardResult)
}


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

fun main(args: Array<String>) {
    sample5_2()
}

private fun sample5_2() {
    val apple = 100
    val appleNum = 2
    val tax = 1.1

    val callGraph = CallGraph(MultiplyLayer(), MultiplyLayer())

    val result = callGraph.forward {
        apple.toDouble() to appleNum.toDouble()
    }.then { results ->
        results[0] to tax
    }.done()

    val backwardResults = callGraph.backward {
        1.toDouble()
    }.then { results ->
        results[0].first
    }.done()

    println(result)
    println(backwardResults)
}

abstract class CallGraphLayer {
    protected lateinit var x: Number
    protected lateinit var y: Number

    abstract fun forward(x: Double, y: Double): Double
    abstract fun backward(dout: Double): Pair<Double, Double>

    operator fun invoke(x: Double, y: Double): Double = this.forward(x, y)
}

class MultiplyLayer : CallGraphLayer() {
    override fun forward(x: Double, y: Double): Double {
        this.x = x
        this.y = y

        return this.x.toDouble() * this.y.toDouble()
    }

    override fun backward(dout: Double): Pair<Double, Double> {
        val dx = dout * this.y.toDouble()
        val dy = dout * this.x.toDouble()

        return dx to dy
    }

}

class CallGraph(private vararg val callGraphLayer: CallGraphLayer) {
    fun forward(getArgument: () -> Pair<Double, Double>): ForwardCallGraphInvocation {
        return ForwardCallGraphInvocation(
                results = listOf(),
                callGraphLayers = this.callGraphLayer.toList()
        ).then { _ -> getArgument() }
    }

    fun backward(getArgument: () -> Double): BackwardCallGraphInvocation {
        return BackwardCallGraphInvocation(
                results = listOf(),
                callGraphLayers = this.callGraphLayer.toList()
        ).then { _ -> getArgument() }
    }
}

class ForwardCallGraphInvocation(private val results: List<Double>,
                                 private val callGraphLayers: List<CallGraphLayer>) {

    fun then(getArgument: (List<Double>) -> Pair<Double, Double>): ForwardCallGraphInvocation {
        val arguments = getArgument(this.results)

        return ForwardCallGraphInvocation(
                this.results + listOf(this.callGraphLayers.first()
                        .forward(arguments.first, arguments.second)),
                this.callGraphLayers.drop(1)
        )
    }

    fun done(): Double = this.results.last()
}

class BackwardCallGraphInvocation(private val results: List<Pair<Double, Double>>,
                                  private val callGraphLayers: List<CallGraphLayer>) {

    fun then(getArgument: (List<Pair<Double, Double>>) -> Double): BackwardCallGraphInvocation {
        val argument = getArgument(this.results)

        return BackwardCallGraphInvocation(
                this.results + listOf(this.callGraphLayers.last().backward(argument)),
                this.callGraphLayers.dropLast(1)
        )
    }

    fun done() = this.results
}
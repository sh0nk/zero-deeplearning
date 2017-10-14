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
package io.github.noriyuki106.call_graph

import io.github.noriyuki106.call_graph.layer.CallGraphLayer
import io.github.noriyuki106.numkt.NumericArray

class CallGraph(private vararg val callGraphLayer: CallGraphLayer) {
    fun forward(getArgument: () -> NumericArray): ForwardCallGraphInvocation {
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

    fun then(getArgument: (List<Double>) -> NumericArray): ForwardCallGraphInvocation {
        val arguments = getArgument(this.results)

        return ForwardCallGraphInvocation(
                this.results + listOf(this.callGraphLayers.first()
                        .forward(arguments)),
                this.callGraphLayers.drop(1)
        )
    }

    fun done(): Double = this.results.last()
}

class BackwardCallGraphInvocation(private val results: List<NumericArray>,
                                  private val callGraphLayers: List<CallGraphLayer>) {

    fun then(getArgument: (List<NumericArray>) -> Double): BackwardCallGraphInvocation {
        val argument = getArgument(this.results)

        return BackwardCallGraphInvocation(
                this.results + listOf(this.callGraphLayers.last().backward(argument)),
                this.callGraphLayers.dropLast(1)
        )
    }

    fun done() = this.results
}
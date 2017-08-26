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
package io.github.noriyuki106.extension

fun ((Double) -> Double).draw(xRange: ClosedFloatingPointRange<Double>, window: Double = 0.2) {
    var targetX = mutableListOf(xRange.start)
    var currentX = xRange.start
    while (currentX <= xRange.endInclusive) {
        currentX += window
        targetX.add(currentX.truncate(2))
    }
    val targetY = targetX.map { this(it).truncate(3) }

    val baseUrl = "https://chart.apis.google.com/chart?chs=300x300&chxt=x,y&cht=lxy&chds=a"
    val xListString = targetX.joinToString(",")
    val yListString = targetY.joinToString(",")
    val graphXRange = (targetX.min() ?: 0 - 0.1) .. (targetX.max() ?: 0 + 0.1)
    val graphYRange = (targetY.min() ?: 0 - 0.1) .. (targetY.max() ?: 0 + 0.1)

    println("$baseUrl&chd=t:$xListString|$yListString&chxr=0,${graphXRange.start},${graphXRange.endInclusive}|1,${graphYRange.start},${graphYRange.endInclusive}")
}

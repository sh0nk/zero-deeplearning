/**
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

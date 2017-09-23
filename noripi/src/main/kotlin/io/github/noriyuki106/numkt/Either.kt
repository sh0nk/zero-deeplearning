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
package io.github.noriyuki106.numkt

class Either<T, R> private constructor(private val left: T?, private val right: R?) {
    companion object {
        fun <T, R> left(value: T): Either<T, R> {
            return Either(value, null)
        }

        fun <T, R> right(value: R): Either<T, R> {
            return Either(null, value)
        }
    }

    fun <U> reduce(leftTransformer: (T) -> U, rightTransformer: (R) -> U): U {
        return this.left?.let(leftTransformer)
                ?: this.right?.let(rightTransformer)
                ?: throw IllegalStateException()
    }
}
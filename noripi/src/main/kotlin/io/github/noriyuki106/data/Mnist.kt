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
package io.github.noriyuki106.data

import io.github.noriyuki106.extension.chunk
import io.github.noriyuki106.extension.toUnsigned
import java.io.File

private const val TRAIN_IMAGE_FILE_PATH = "data/mnist/train-images-idx3-ubyte"
private const val TRAIN_LABEL_FILE_PATH = "data/mnist/train-labels-idx1-ubyte"
private const val ROW_SIZE = 28
private const val ONE_DATA_SIZE = ROW_SIZE * ROW_SIZE
private const val DATA_SIZE = 60000

class Mnist {
    private val imageContent: ByteArray = File(TRAIN_IMAGE_FILE_PATH).readBytes()
            .drop(16)
            .toByteArray()

    private val labelContent: ByteArray = File(TRAIN_LABEL_FILE_PATH).readBytes()
            .drop(8)
            .toByteArray()

    operator fun get(i: Int) = MnistData.instanceAt(i) ?: MnistData(
            image = this.imageContent.sliceArray(ONE_DATA_SIZE * i .. ONE_DATA_SIZE * (i + 1) - 1),
            label = this.labelContent[i],
            index = i
    )
}

class MnistData(private val image: ByteArray, private val label: Byte, private val index: Int) {
    init {
        cachedData[index] = this
    }

    companion object {
        private var cachedData = arrayOfNulls<MnistData>(DATA_SIZE)

        fun instanceAt(i: Int): MnistData? {
            return cachedData[i]
        }
    }

    fun print() {
        println("label: ${this.label}")
        this.image.forEachIndexed { idx, value ->
            print(when (value.toUnsigned()) {
                in 0..63 -> " "
                in 64..127 -> "."
                in 128..191 -> "+"
                else -> "*"
            })
            if ((idx + 1) % ROW_SIZE == 0) println()
        }
    }

}

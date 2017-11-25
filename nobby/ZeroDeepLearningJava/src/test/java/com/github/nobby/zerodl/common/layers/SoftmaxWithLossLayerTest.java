package com.github.nobby.zerodl.common.layers;

import org.jblas.DoubleMatrix;
import org.junit.Assert;
import org.junit.Test;

import static org.junit.Assert.*;

public class SoftmaxWithLossLayerTest {
    @Test
    public void test_forward() {
        double[][] dataArr = {
                {-0.002946, 0.000544, -0.009666, -0.003741, -0.002475, -0.001017, 0.001615, -0.004701, -0.013769, -0.001414},
                {-0.002946, 0.000544, -0.009666, -0.003741, -0.002475, -0.001017, 0.001615, -0.004701, -0.013769, -0.001414},
                {-0.002946, 0.000544, -0.009666, -0.003741, -0.002475, -0.001017, 0.001615, -0.004701, -0.013769, -0.001414}
        };
        double[][] labelArr = {
                {0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000},
                {0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000},
                {0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000}
        };
        SoftmaxWithLossLayer softmaxWithLossLayer = new SoftmaxWithLossLayer();
        double loss = softmaxWithLossLayer.forward(new DoubleMatrix(dataArr), new DoubleMatrix(labelArr));
        Assert.assertEquals(2.3, loss, 0.1);
    }
}
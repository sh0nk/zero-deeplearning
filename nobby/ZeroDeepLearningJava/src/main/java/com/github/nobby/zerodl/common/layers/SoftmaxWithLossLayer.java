package com.github.nobby.zerodl.common.layers;

import com.github.nobby.zerodl.common.Functions;
import org.jblas.DoubleMatrix;

public class SoftmaxWithLossLayer {
    private double loss;
    private DoubleMatrix y; // output of softmax function
    private DoubleMatrix t; // labelData
    public SoftmaxWithLossLayer() {}

    public double forward(DoubleMatrix x, DoubleMatrix t) {
        this.t = t;
        y = Functions.softmax(x);
        loss = Functions.crossEntropyError(y, t);

        return loss;
    }

    public DoubleMatrix backward() {
        double batch_size = (double) t.rows;
        DoubleMatrix dx = y.sub(t);
        return dx.div(batch_size);
    }
}

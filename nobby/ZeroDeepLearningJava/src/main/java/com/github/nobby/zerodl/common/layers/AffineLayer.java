package com.github.nobby.zerodl.common.layers;

import org.jblas.DoubleMatrix;

public class AffineLayer {
    private DoubleMatrix W;
    private DoubleMatrix b;
    private DoubleMatrix x;
    private DoubleMatrix dW;
    private DoubleMatrix db;

    AffineLayer() {}

    public DoubleMatrix forward(DoubleMatrix x) {
        this.x = x;
        DoubleMatrix out = x.mmul(this.W);
        return out;
    }
}

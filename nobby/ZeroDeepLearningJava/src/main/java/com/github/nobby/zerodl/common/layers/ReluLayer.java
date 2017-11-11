package com.github.nobby.zerodl.common.layers;

import org.jblas.DoubleMatrix;

public class ReluLayer {
    private DoubleMatrix maskMatrix;

    public DoubleMatrix forward(DoubleMatrix x) {
        maskMatrix = x.gt(0);
        DoubleMatrix out = x.dup();
        return out.mmul(maskMatrix);
    }

    public DoubleMatrix backward(DoubleMatrix dout) {
        return dout.mmul(maskMatrix);
    }
}

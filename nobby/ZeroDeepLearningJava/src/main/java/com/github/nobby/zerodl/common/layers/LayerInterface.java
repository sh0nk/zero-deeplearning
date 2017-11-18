package com.github.nobby.zerodl.common.layers;

import org.jblas.DoubleMatrix;

public interface LayerInterface {
    DoubleMatrix forward(DoubleMatrix x);
    DoubleMatrix backward(DoubleMatrix dout);
}

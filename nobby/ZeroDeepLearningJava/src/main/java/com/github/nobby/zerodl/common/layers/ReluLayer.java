package com.github.nobby.zerodl.common.layers;

import com.github.nobby.zerodl.dataset.MnistHandler;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ReluLayer implements LayerInterface {
    private static final Logger logger = LoggerFactory.getLogger(MnistHandler.class);

    private DoubleMatrix maskMatrix;

    public DoubleMatrix forward(DoubleMatrix x) {
        maskMatrix = x.gt(0);

        DoubleMatrix out = x.dup();
        return out.mul(maskMatrix);
    }

    public DoubleMatrix backward(DoubleMatrix dout) {
        dout = dout.mul(maskMatrix);
        return dout;
    }
}

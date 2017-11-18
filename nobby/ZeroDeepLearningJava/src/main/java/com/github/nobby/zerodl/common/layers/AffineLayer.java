package com.github.nobby.zerodl.common.layers;

import lombok.Data;
import org.jblas.DoubleMatrix;

@Data
public class AffineLayer implements LayerInterface {
    private DoubleMatrix W;
    private DoubleMatrix b;
    private DoubleMatrix x;
    private DoubleMatrix dW;
    private DoubleMatrix db;

    public AffineLayer(DoubleMatrix w, DoubleMatrix b) {
        this.W = w;
        this.b = b;
    }

    public DoubleMatrix forward(DoubleMatrix x) {
        this.x = x;
        DoubleMatrix out = x.mmul(this.W).addRowVector(this.b);
        return out;
    }

    public DoubleMatrix backward(DoubleMatrix dout) {
        DoubleMatrix dx = dout.mmul(this.W.transpose());
        this.dW = this.x.transpose().mmul(dout);
        this.db = dout.columnSums();
        return dx;
    }
}

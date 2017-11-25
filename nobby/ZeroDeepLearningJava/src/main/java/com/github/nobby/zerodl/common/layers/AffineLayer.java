package com.github.nobby.zerodl.common.layers;

import com.github.nobby.zerodl.dataset.MnistHandler;
import lombok.Data;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

@Data
public class AffineLayer implements LayerInterface {
    private static final Logger logger = LoggerFactory.getLogger(MnistHandler.class);

    private DoubleMatrix W;
    private DoubleMatrix B;
    private DoubleMatrix X;
    private DoubleMatrix dW;
    private DoubleMatrix dB;

    public AffineLayer(DoubleMatrix w, DoubleMatrix b) {
        this.W = w;
        this.B = b;
    }

    public DoubleMatrix forward(DoubleMatrix x) {
        this.X = x;
        DoubleMatrix out = x.mmul(this.W).addRowVector(this.B);

        return out;
    }

    public DoubleMatrix backward(DoubleMatrix dout) {
        DoubleMatrix dx = dout.mmul(this.W.transpose());
        this.dW = this.X.transpose().mmul(dout);
        this.dB = dout.columnSums();
        return dx;
    }
}

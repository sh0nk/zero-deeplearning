package com.github.nobby.zerodl.chap5;

import com.github.nobby.zerodl.common.Functions;
import com.github.nobby.zerodl.common.layers.AffineLayer;
import com.github.nobby.zerodl.common.layers.LayerInterface;
import com.github.nobby.zerodl.common.layers.ReluLayer;
import com.github.nobby.zerodl.common.layers.SoftmaxWithLossLayer;
import lombok.Data;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

/**
 * Created by onishinobuhiro on 2017/09/24.
 * Two Layer Neural Network
 */
@Data
public class TwoLayerNet {
    private static final Logger logger = LoggerFactory.getLogger(TwoLayerNet.class);

    private double weightInitStd = 0.01;
    private DoubleMatrix W1;
    private DoubleMatrix B1;
    private DoubleMatrix W2;
    private DoubleMatrix B2;

    private List<LayerInterface> layerList;
    private SoftmaxWithLossLayer lastLayer;

    TwoLayerNet(int input_size, int hidden_size, int output_size) {
        W1 = new DoubleMatrix(createWeightMatrix(input_size, hidden_size));
        double[] arrayB1 = new double[hidden_size];
        Arrays.fill(arrayB1, 0);
        B1 = new DoubleMatrix(arrayB1).transpose();

        W2 = new DoubleMatrix(createWeightMatrix(hidden_size, output_size));
        double[] arrayB2 = new double[output_size];
        Arrays.fill(arrayB2, 0);
        B2 = new DoubleMatrix(arrayB2).transpose();

        layerList = new ArrayList<LayerInterface>();
        initLayers();
    }

    private void initLayers() {
        layerList.add(new AffineLayer(W1, B1));
        layerList.add(new ReluLayer());
        layerList.add(new AffineLayer(W2, B2));
        lastLayer = new SoftmaxWithLossLayer();
    }

    public void updLayerParams() {
        layerList.clear();
        initLayers();
    }

    /**
     * @param x : image data. 100 rows * 784 columns
     */
    private DoubleMatrix predict(DoubleMatrix x) {
        for (LayerInterface layer : layerList) {
            x = layer.forward(x);
        }
        return x;
    }

    public double loss(DoubleMatrix x, DoubleMatrix t) {
        DoubleMatrix y = predict(x);
        return lastLayer.forward(y, t);
    }

    public float accuracy(DoubleMatrix x, DoubleMatrix t) {
        int correctCount = 0;
        DoubleMatrix y = predict(x);
        for (int i = 0; i < y.rows; i++) {
            int predictValue = y.getRow(i).argmax();
            int labelValue = t.getRow(i).argmax();
            if (predictValue == labelValue) {
                correctCount++;
            }
        }
        return (float) correctCount / y.rows;
    }

    public Gradient gradient(DoubleMatrix x, DoubleMatrix t) {
        // forward
        loss(x, t);

        // backward
        DoubleMatrix dout = lastLayer.backward();
        List<LayerInterface> reverseLayerList = new ArrayList<>(layerList);
        Collections.reverse(reverseLayerList);
        for (LayerInterface layer : reverseLayerList) {
            dout = layer.backward(dout);
        }

        Gradient gradient = new Gradient();
        AffineLayer affineLayer1 = (AffineLayer) layerList.get(0);
        AffineLayer affineLayer2 = (AffineLayer) layerList.get(2);

        gradient.setW1(affineLayer1.getDW());
        gradient.setB1(affineLayer1.getDB());
        gradient.setW2(affineLayer2.getDW());
        gradient.setB2(affineLayer2.getDB());

        return gradient;
    }

    private double[][] createWeightMatrix(int rowNum, int columnNum) {
        double[][] w = new double[rowNum][columnNum];
        Random random = new Random();
        for (int i = 0; i < rowNum; i++) {
            for (int j = 0; j < columnNum; j++) {
                w[i][j] = weightInitStd * random.nextGaussian();
            }
        }
        return w;
    }
}

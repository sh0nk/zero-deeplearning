package com.github.nobby.zerodl.chap4;

import com.github.nobby.zerodl.common.Functions;
import com.github.nobby.zerodl.dataset.Label;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

/**
 * Created by onishinobuhiro on 2017/09/24.
 * Two Layer Neural Network
 */
public class TwoLayerNet {
    private static final Logger logger = LoggerFactory.getLogger(TwoLayerNet.class);
    private static final double h = 0.0001;

    private double weightInitStd = 0.01;
    private DoubleMatrix W1;
    private DoubleMatrix B1;
    private DoubleMatrix W2;
    private DoubleMatrix B2;

    TwoLayerNet(int input_size, int hidden_size, int output_size) {
        W1 = new DoubleMatrix(createWeightMatrix(input_size, hidden_size));
        double[] arrayB1 = new double[hidden_size];
        Arrays.fill(arrayB1, 0);
        B1 = new DoubleMatrix(arrayB1).transpose();

        W2 = new DoubleMatrix(createWeightMatrix(hidden_size, output_size));
        double[] arrayB2 = new double[output_size];
        Arrays.fill(arrayB2, 0);
        B2 = new DoubleMatrix(arrayB2).transpose();
    }

    TwoLayerNet(DoubleMatrix w1, DoubleMatrix b1, DoubleMatrix w2, DoubleMatrix b2) {
        W1 = w1; W2 = w2;
        B1 = b1; B2 = b2;
    }

    /**
     * @param x : image data. 100 rows * 784 columns
     */
    private DoubleMatrix predict(DoubleMatrix x) {
        DoubleMatrix a1 = x.mmul(W1);
        // add b1 matrix
        for (int i = 0; i < a1.rows; i++) {
            for (int j = 0; j < a1.columns; j++) {
                a1.put(i, j, a1.get(i, j) + B1.get(0, j));
            }
        }
        DoubleMatrix z1 = Functions.sigmoid(a1);

        // add b2 matrix
        DoubleMatrix a2 = z1.mmul(W2);
        for (int i = 0; i < a2.rows; i++) {
            for (int j = 0; j < a2.columns; j++) {
                a2.put(i, j, a2.get(i, j) + B2.get(0, j));
            }
        }
        return Functions.softmax(a2);
    }

    public double loss(DoubleMatrix x, ArrayList<Label> t) {
        DoubleMatrix y = predict(x);
        return Functions.crossEntropyError(x, t);
    }

    public float accuracy(DoubleMatrix x, ArrayList<Label> t) {
        int correctCount = 0;
        DoubleMatrix y = predict(x);
        for (int i = 0; i < y.rows; i++) {
            int predictValue = y.getRow(i).argmax();
            if (predictValue == t.get(i).getLabelValue()) {
                correctCount++;
            }
        }
        return correctCount / y.rows;
    }

    public void numericalGradient(DoubleMatrix x, ArrayList<Label> t, double learningRatio) {
        logger.info("calculate gradient W1...");
        DoubleMatrix gradW1 = numericalGradientW1(x, t);
        logger.info("calculate gradient B1...");
        DoubleMatrix gradB1 = numericalGradientB1(x, t);
        logger.info("calculate gradient W2...");
        DoubleMatrix gradW2 = numericalGradientW2(x, t);
        logger.info("calculate gradient B2...");
        DoubleMatrix gradB2 = numericalGradientB2(x, t);

        W1 = W1.sub(gradW1.mul(learningRatio));
        B1 = B1.sub(gradB1.mul(learningRatio));
        W2 = W2.sub(gradW2.mul(learningRatio));
        B2 = B2.sub(gradB2.mul(learningRatio));
    }

    private DoubleMatrix numericalGradientW1(DoubleMatrix x, ArrayList<Label> t) {
        DoubleMatrix gradW1 = DoubleMatrix.zeros(W1.rows, W1.columns);
        for (int i = 0; i < W1.rows; i++) {
            for (int j = 0; j < W1.columns; j++) {
                DoubleMatrix tmpW1 = W1.dup();
                tmpW1.put(i, j, W1.get(i, j) + h);
                TwoLayerNet twoLayerNet = new TwoLayerNet(tmpW1, B1, W2, B2);
                double fd1 = twoLayerNet.loss(x, t);

                tmpW1.put(i, j, W1.get(i, j) - h);
                double fd2 = twoLayerNet.loss(x, t);
                gradW1.put(i, j, ((fd1 - fd2) / 2 * h));
            }
        }
        return gradW1;
    }

    private DoubleMatrix numericalGradientW2(DoubleMatrix x, ArrayList<Label> t) {
        DoubleMatrix gradW2 = DoubleMatrix.zeros(W2.rows, W2.columns);
        for (int i = 0; i < W2.rows; i++) {
            for (int j = 0; j < W2.columns; j++) {
                DoubleMatrix tmpW2 = W2.dup();
                tmpW2.put(i, j, W2.get(i, j) + h);
                TwoLayerNet twoLayerNet = new TwoLayerNet(W1, B1, tmpW2, B2);
                double fd1 = twoLayerNet.loss(x, t);

                tmpW2.put(i, j, W2.get(i, j) - h);
                double fd2 = twoLayerNet.loss(x, t);
                gradW2.put(i, j, ((fd1 - fd2) / 2 * h));
            }
        }
        return gradW2;
    }

    private DoubleMatrix numericalGradientB1(DoubleMatrix x, ArrayList<Label> t) {
        DoubleMatrix gradB1 = DoubleMatrix.zeros(B1.rows, B1.columns);
        for (int i = 0; i < B1.rows; i++) {
            for (int j = 0; j < B1.columns; j++) {
                DoubleMatrix tmpB1 = B1.dup();
                tmpB1.put(i, j, B1.get(i, j) + h);
                TwoLayerNet twoLayerNet = new TwoLayerNet(W1, tmpB1, W2, B2);
                double fd1 = twoLayerNet.loss(x, t);

                tmpB1.put(i, j, B1.get(i, j) - h);
                double fd2 = twoLayerNet.loss(x, t);
                gradB1.put(i, j, ((fd1 - fd2) / 2 * h));
            }
        }
        return gradB1;
    }

    private DoubleMatrix numericalGradientB2(DoubleMatrix x, ArrayList<Label> t) {
        DoubleMatrix gradB2 = DoubleMatrix.zeros(B2.rows, B2.columns);
        for (int i = 0; i < B2.rows; i++) {
            for (int j = 0; j < B2.columns; j++) {
                DoubleMatrix tmpB2 = B2.dup();
                tmpB2.put(i, j, B2.get(i, j) + h);
                TwoLayerNet twoLayerNet = new TwoLayerNet(W1, B1, W2, tmpB2);
                double fd1 = twoLayerNet.loss(x, t);

                tmpB2.put(i, j, B2.get(i, j) - h);
                double fd2 = twoLayerNet.loss(x, t);
                gradB2.put(i, j, ((fd1 - fd2) / 2 * h));
            }
        }
        return gradB2;
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

package com.github.nobby.zerodl.chap5;

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

        DoubleMatrix a2 = z1.mmul(W2);
        // add b2 matrix
        for (int i = 0; i < a2.rows; i++) {
            for (int j = 0; j < a2.columns; j++) {
                a2.put(i, j, a2.get(i, j) + B2.get(0, j));
            }
        }
        return Functions.softmax(a2);
    }

    public double loss(DoubleMatrix x, DoubleMatrix t) {
        DoubleMatrix y = predict(x);
        return Functions.crossEntropyError(x, t);
    }

    public float accuracy(DoubleMatrix x, DoubleMatrix t) {
        int correctCount = 0;
        DoubleMatrix y = predict(x);
        for (int i = 0; i < y.rows; i++) {
            int predictValue = y.getRow(i).argmax();
            //TODO 直す
            /*
            if (predictValue == t.get(i).getLabelValue()) {
                correctCount++;
            }
            */
        }
        return correctCount / y.rows;
    }

    public Gradient gradient(DoubleMatrix x, ArrayList<Label> t) {
        // forward
        DoubleMatrix a1 = x.mmul(W1);
        for (int i = 0; i < a1.rows; i++) {
            for (int j = 0; j < a1.columns; j++) {
                a1.put(i, j, a1.get(i, j) + B1.get(0, j));
            }
        }
        DoubleMatrix z1 = Functions.sigmoid(a1);
        DoubleMatrix a2 = z1.mmul(W2);
        for (int i = 0; i < a2.rows; i++) {
            for (int j = 0; j < a2.columns; j++) {
                a2.put(i, j, a2.get(i, j) + B2.get(0, j));
            }
        }
        DoubleMatrix y = Functions.softmax(a2);

        // backward
        DoubleMatrix tMatrix = new DoubleMatrix(t.size(), t.get(0).getLabel().columns);
        for (int i = 0; i < t.size(); i++) {
            tMatrix = DoubleMatrix.concatVertically(tMatrix, t.get(i).getLabel());
        }
        DoubleMatrix dy = y.sub(tMatrix).div(x.rows);

        Gradient gradient = new Gradient();
        gradient.setW2(z1.transpose().mmul(dy));
        gradient.setB2(dy.rowSums());

        DoubleMatrix da1 = dy.mmul(W2.transpose());
        DoubleMatrix dz1 = Functions.sigmoidGrad(a1).mmul(da1);



        //W1 = W1.sub(gradW1.mul(learningRatio));
        //B1 = B1.sub(gradB1.mul(learningRatio));
        //W2 = W2.sub(gradW2.mul(learningRatio));
        //B2 = B2.sub(gradB2.mul(learningRatio));


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

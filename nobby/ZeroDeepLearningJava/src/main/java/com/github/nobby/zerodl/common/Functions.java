package com.github.nobby.zerodl.common;

import org.jblas.DoubleMatrix;

/**
 * Created by onishinobuhiro on 2017/09/27.
 */
public class Functions {
    private static final double EPSILON = 0.0000001;

    public static DoubleMatrix sigmoid(DoubleMatrix input) {
        for (int i = 0; i < input.rows; i++) {
            for (int j = 0; j < input.columns; j++) {
                input.put(i, j, 1 / (1 + Math.exp(-input.get(i,j))));
            }
        }
        return input;
    }

    public static DoubleMatrix sigmoidGrad(DoubleMatrix input) {
        DoubleMatrix oneMatrix = DoubleMatrix.ones(input.rows, input.columns);
        return oneMatrix.sub(sigmoid(input)).mmul(input);
    }

    public static DoubleMatrix softmax(DoubleMatrix input) {
        // for matrix
        DoubleMatrix x = input.transpose();
        x = x.subRowVector(x.columnMaxs());
        DoubleMatrix expMatrix = x.dup();
        for (int i = 0; i < x.rows; i++) {
            for (int j = 0; j < x.columns; j++) {
                double exp = Math.exp(x.get(i, j));
                expMatrix.put(i, j, exp);
            }
        }
        DoubleMatrix y = expMatrix.divRowVector(expMatrix.columnSums());
        return y.transpose();

        /*
        double max_x = input.max();
        DoubleMatrix x = input.sub(max_x);  // counter measure for overflow
        //DoubleMatrix x = input.sub(max_x);  // counter measure for overflow
        double sum_exp_x = 0;
        for (int i = 0; i < input.rows; i++) {
            for (int j = 0; j < input.columns; j++) {
                sum_exp_x += Math.exp(x.get(i, j));
                x.put(i, j, Math.exp(x.get(i, j)));
            }
        }
        logger.info("sum_exp_x: {}", sum_exp_x);
        return x.div(sum_exp_x);
        */
    }

    /**
     * @param y neural network output
     * @param t train labels
     */
    public static double crossEntropyError(DoubleMatrix y, DoubleMatrix t) {
        double errorSum = 0;
        for (int i = 0; i < y.rows; i++) {
            double yError = y.get(i, getLabelIndex(t, i)) + EPSILON;
            errorSum += Math.log(yError);
        }
        return -errorSum / (double) y.rows;
    }

    /**
     * get the label index from 'one-hot-label' format label data
     * @param t labelData Matrix
     * @param targetRow get the label index of this row number
     * @return
     */
    public static int getLabelIndex(DoubleMatrix t, int targetRow) {
        DoubleMatrix label = t.getRow(targetRow);
        return label.argmax();
    }


    public static void printMatrix(DoubleMatrix x) {
        int max = x.rows > 100 ? 100 : x.rows;
        for (int i = 0; i < max; i++) {
            x.getRow(i).print();
        }
    }


}

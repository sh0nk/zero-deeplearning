package com.github.nobby.zerodl.common;

import com.github.nobby.zerodl.dataset.Label;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Arrays;

/**
 * Created by onishinobuhiro on 2017/09/27.
 */
public class Functions {
    private static final double EPSILON = 0.0000001;
    private static final Logger logger = LoggerFactory.getLogger(Functions.class);

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
        DoubleMatrix result = new DoubleMatrix();
        for (int i = 0; i < input.rows; i++) {
            DoubleMatrix targetRow = input.getRow(i);
            double max_a = targetRow.max();
            double[] exp_a = new double[targetRow.columns];
            double sum_exp_a = 0;
            for (int j = 0; j < input.columns; j++) {
                // counter measure for overflow
                exp_a[j] = Math.exp(targetRow.get(j) - max_a);
                sum_exp_a += exp_a[j];
            }
            final double f_sum_exp_a = sum_exp_a;
            Arrays.stream(exp_a).map(e ->  e / f_sum_exp_a);
            result = i == 0 ? new DoubleMatrix(exp_a).transpose() : DoubleMatrix.concatVertically(result, new DoubleMatrix(exp_a).transpose());
        }
        return result;
    }

    /**
     * @param y neural network output
     * @param t train labels
     */
    public static double crossEntropyError(DoubleMatrix y, DoubleMatrix t) {
        double error_sum = 0;
        for (int i = 0; i < y.rows; i++) {
            double y_value = y.getRow(i).get(getLabelValue(t, i));
            double error = Math.log(y_value + EPSILON);
            error_sum += error;
        }
        return -1 * error_sum / y.rows;
    }

    /**
     * get the label value from 'one-hot-label' format label data
     * @param t labelData Matrix
     * @param targetRow get the label value of this row number
     * @return
     */
    public static int getLabelValue(DoubleMatrix t, int targetRow) {
        DoubleMatrix label = t.getRow(targetRow);
        return label.argmax();
    }
}

package com.github.nobby.zerodl.dataset;


import lombok.Data;
import org.jblas.DoubleMatrix;

/**
 * Created by onishinobuhiro on 2017/09/23.
 */
@Data
public class MnistData {
    private DoubleMatrix trainData;       // 60000 rows * 784 columns matrix
    private DoubleMatrix trainLabels;     // 60000 rows * 10  columns matrix
    private DoubleMatrix testData;        // 10000 rows * 784 columns matrxi
    private DoubleMatrix testLabels;      // 10000 rows * 10  columns matrix

    MnistData(double[][] trainData, double[][] trainLabels, double[][] testData, double[][] testLabels) {
        this.trainData = new DoubleMatrix(trainData);
        this.trainLabels = new DoubleMatrix(trainLabels);
        this.testData = new DoubleMatrix(testData);
        this.testLabels = new DoubleMatrix(testLabels);
    }

    public BatchMnistData getTrainData4Batch(int size) {
        int trainDataNum = trainData.rows;
        int rand = (int)(Math.random() * trainDataNum);
        DoubleMatrix batchData = trainData.getRow(rand);
        DoubleMatrix batchLabels = trainLabels.getRow(rand);

        for (int i = 1; i < size; i++) {
            rand = (int)(Math.random() * trainDataNum);
            batchData = DoubleMatrix.concatVertically(batchData, trainData.getRow(rand));
            batchLabels = DoubleMatrix.concatVertically(batchLabels, trainLabels.getRow(rand));
        }
        return new BatchMnistData(batchData, batchLabels);
    }
}

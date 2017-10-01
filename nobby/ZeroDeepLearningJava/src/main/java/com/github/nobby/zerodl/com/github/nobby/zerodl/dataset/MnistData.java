package com.github.nobby.zerodl.com.github.nobby.zerodl.dataset;


import org.jblas.DoubleMatrix;

import java.util.ArrayList;

/**
 * Created by onishinobuhiro on 2017/09/23.
 */

public class MnistData {
    private DoubleMatrix trainData;       // 60000 rows * 784 columns matrix
    private ArrayList<Label> trainLabels; // 60000 rows * 10  columns matrix
    private DoubleMatrix testData;        // 10000 rows * 784 columns matrxi
    private ArrayList<Label> testLabels;  // 10000 rows * 10  columns matrix

    MnistData() {}
    MnistData(double[][] trainData, ArrayList<Label> trainLabels, double[][] testData, ArrayList<Label> testLabels) {
        this.trainData = new DoubleMatrix(trainData);
        this.trainLabels = trainLabels;
        this.testData = new DoubleMatrix(testData);
        this.testLabels = testLabels;
    }

    public void setTrainData(DoubleMatrix trainData){this.trainData = trainData; }
    public DoubleMatrix getTrainData() { return this.trainData; }

    public void setTrainLabels(ArrayList<Label> trainLabels){this.trainLabels = trainLabels; }
    public ArrayList<Label> getTrainLabels() { return this.trainLabels; }

    public void setTestData(DoubleMatrix testData){this.testData = testData; }
    public DoubleMatrix getTestData() { return this.testData; }

    public void setTestLabels(ArrayList<Label> testLabels){this.testLabels = testLabels; }
    public ArrayList<Label> getTestLabels() { return this.testLabels; }

    public BatchMnistData getTrainData4Batch(int size) {
        int trainDataNum = trainData.rows;
        int rand = (int)(Math.random() * trainDataNum);
        DoubleMatrix batchData = trainData.getRow(rand);
        ArrayList<Label> batchLabelList = new ArrayList<>();
        batchLabelList.add(trainLabels.get(rand));

        for (int i = 1; i < size; i++) {
            rand = (int)(Math.random() * trainDataNum);
            batchData = DoubleMatrix.concatVertically(batchData, trainData.getRow(rand));
            batchLabelList.add(trainLabels.get(rand));
        }
        return new BatchMnistData(batchData, batchLabelList);
    }
}

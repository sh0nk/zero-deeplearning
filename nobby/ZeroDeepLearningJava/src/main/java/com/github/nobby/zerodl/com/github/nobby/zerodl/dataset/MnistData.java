package com.github.nobby.zerodl.com.github.nobby.zerodl.dataset;

/**
 * Created by onishinobuhiro on 2017/09/23.
 */

public class MnistData {
    private double[][] trainData;
    private int[][] trainLabel;
    private double[][] testData;
    private int[][] testLabel;

    MnistData() {}
    MnistData(double[][] trainData, int[][] trainLabel, double[][] testData, int[][] testLabel) {
        this.trainData = trainData;
        this.trainLabel = trainLabel;
        this.testData = testData;
        this.testLabel = testLabel;
    }

    public void setTrainData(double[][] trainData){this.trainData = trainData; }
    public double[][] getTrainData() { return this.trainData; }

    public void setTrainLabel(int[][] trainLabel){this.trainLabel = trainLabel; }
    public int[][] getTrainLabel() { return this.trainLabel; }

    public void setTestData(double[][] testData){this.testData = testData; }
    public double[][] getTestData() { return this.testData; }

    public void setTestLabel(int[][] testLabel){this.testLabel = testLabel; }
    public int[][] getTestLabel() { return this.testLabel; }


}

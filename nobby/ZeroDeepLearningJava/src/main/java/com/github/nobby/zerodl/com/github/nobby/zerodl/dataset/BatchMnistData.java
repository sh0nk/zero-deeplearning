package com.github.nobby.zerodl.com.github.nobby.zerodl.dataset;

import org.jblas.DoubleMatrix;

import java.util.ArrayList;

/**
 * Created by onishinobuhiro on 2017/10/01.
 */
public class BatchMnistData {
    DoubleMatrix batchData;
    ArrayList<Label> batchLabels;

    BatchMnistData(DoubleMatrix batchData, ArrayList<Label> batchLabel) {
        this.batchData = batchData;
        this.batchLabels = batchLabel;
    }

    public DoubleMatrix getBatchData() {
        return this.batchData;
    }

    public ArrayList<Label> getBatchLabel() {
        return this.batchLabels;
    }
}

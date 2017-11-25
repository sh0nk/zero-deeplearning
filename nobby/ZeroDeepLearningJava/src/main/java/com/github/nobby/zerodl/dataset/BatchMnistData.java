package com.github.nobby.zerodl.dataset;

import lombok.Data;
import org.jblas.DoubleMatrix;

import java.util.ArrayList;

/**
 * Created by onishinobuhiro on 2017/10/01.
 */
@Data
public class BatchMnistData {
    DoubleMatrix batchData;
    DoubleMatrix batchLabels;

    BatchMnistData(DoubleMatrix batchData, DoubleMatrix batchLabels) {
        this.batchData = batchData;
        this.batchLabels = batchLabels;
    }
}

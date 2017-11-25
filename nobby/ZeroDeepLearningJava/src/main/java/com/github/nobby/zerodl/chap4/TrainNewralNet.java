package com.github.nobby.zerodl.chap4;

import com.github.nobby.zerodl.dataset.BatchMnistData;
import com.github.nobby.zerodl.dataset.MnistData;
import com.github.nobby.zerodl.dataset.MnistHandler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by onishinobuhiro on 2017/09/23.
 */
public class TrainNewralNet {
    private static final Logger logger = LoggerFactory.getLogger(TrainNewralNet.class);
    
    public static void main(String[] args) {
        List trainLossList = new ArrayList();
        List trainAccList = new ArrayList();
        List testAccList = new ArrayList();

        MnistHandler mnistHandler = new MnistHandler();
        try {
            MnistData mnistData = mnistHandler.loadMnist(true);
            TwoLayerNet twoLayerNet = new TwoLayerNet(784, 50, 10);

            int itersNum = 10000;
            int trainSize = mnistData.getTrainData().length;
            int batchSize = 100;
            double learningRate = 0.1;
            int iter_per_epoch = Math.max(trainSize / batchSize, 1);

            for (int i = 0; i < itersNum; i++) {
                logger.info("iters count: {}", i);
                BatchMnistData batchMnistData = mnistData.getTrainData4Batch(batchSize);
                twoLayerNet.numericalGradient(batchMnistData.getBatchData(), batchMnistData.getBatchLabels(), learningRate);
                double loss = twoLayerNet.loss(batchMnistData.getBatchData(), batchMnistData.getBatchLabels());
                trainLossList.add(loss);
                logger.info("  loss value: {}", loss);

                if (i % iter_per_epoch == 0) {
                    float trainAcc = twoLayerNet.accuracy(mnistData.getTrainData(), mnistData.getTrainLabels());
                    float testAcc = twoLayerNet.accuracy(mnistData.getTestData(), mnistData.getTestLabels());

                    trainAccList.add(trainAcc);
                    testAccList.add(testAcc);
                }
            }
        } catch (IOException e) {
            logger.error("load MNIST DATA failed.", e);
        }
    }
}

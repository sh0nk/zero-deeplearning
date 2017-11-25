package com.github.nobby.zerodl.chap5;

import com.github.nobby.zerodl.dataset.BatchMnistData;
import com.github.nobby.zerodl.dataset.MnistData;
import com.github.nobby.zerodl.dataset.MnistHandler;
import com.github.sh0nk.matplotlib4j.Plot;
import com.github.sh0nk.matplotlib4j.PythonExecutionException;
import org.jblas.DoubleMatrix;
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
            int trainSize = mnistData.getTrainData().rows;
            int batchSize = 100;
            double learningRate = 0.1;
            int iter_per_epoch = Math.max(trainSize / batchSize, 1);
            logger.info("iter_per_epoch : {}", iter_per_epoch);

            for (int i = 0; i < itersNum; i++) {
                BatchMnistData batchMnistData = mnistData.getTrainData4Batch(batchSize);
                DoubleMatrix xBatch = batchMnistData.getBatchData();
                DoubleMatrix tBatch = batchMnistData.getBatchLabels();

                Gradient gradient = twoLayerNet.gradient(xBatch, tBatch);
                // update
                twoLayerNet.setW1(twoLayerNet.getW1().sub( (gradient.getW1().mul(learningRate))) );
                twoLayerNet.setB1(twoLayerNet.getB1().sub( (gradient.getB1().mul(learningRate))) );
                twoLayerNet.setW2(twoLayerNet.getW2().sub( (gradient.getW2().mul(learningRate))) );
                twoLayerNet.setB2(twoLayerNet.getB2().sub( (gradient.getB2().mul(learningRate))) );

                twoLayerNet.updLayerParams();

                double loss = twoLayerNet.loss(xBatch, tBatch);

                trainLossList.add(loss);

                if (i % iter_per_epoch == 0) {
                    float trainAcc = twoLayerNet.accuracy(mnistData.getTrainData(), mnistData.getTrainLabels());
                    float testAcc = twoLayerNet.accuracy(mnistData.getTestData(), mnistData.getTestLabels());

                    trainAccList.add(trainAcc);
                    testAccList.add(testAcc);
                    logger.info("  loss value: {}", loss);
                    logger.info("  train Acc value: {}", trainAcc);
                    logger.info("  testAcc value: {}", testAcc);
                }
            }

            Plot plt = Plot.create();
            plt.plot()
                    .add(trainAccList)
                    .label("train acc");
            plt.plot()
                    .add(testAccList)
                    .label("test acc")
                    .linestyle("--");
            plt.xlabel("epochs");
            plt.ylabel("accuracy");
            plt.title("title");
            plt.legend().loc("lower right");
            plt.show();

        } catch (IOException e) {
            logger.error("load MNIST DATA failed.", e);
        } catch (PythonExecutionException e) {
            logger.error("python env error", e);
        }
    }
}

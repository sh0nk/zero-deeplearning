package com.github.nobby.zerodl.com.github.nobby.zerodl.chap4;

import com.github.nobby.zerodl.com.github.nobby.zerodl.dataset.MnistData;
import com.github.nobby.zerodl.com.github.nobby.zerodl.dataset.MnistHandler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

/**
 * Created by onishinobuhiro on 2017/09/23.
 */
public class TrainNewralNet {
    private static final Logger logger = LoggerFactory.getLogger(MnistHandler.class);

    MnistHandler mnistHandler = new MnistHandler();

    public void exec() {
        try {
            MnistData mnistData = mnistHandler.loadMnist(true);
        } catch (IOException e) {
            logger.error("load MNIST DATA failed.", e);
        }
    }
}

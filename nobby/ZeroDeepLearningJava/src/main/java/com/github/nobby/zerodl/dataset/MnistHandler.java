package com.github.nobby.zerodl.dataset;

import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.*;
import java.net.URL;
import java.net.URLConnection;
import java.util.ArrayList;
import java.util.zip.GZIPInputStream;

/**
 * Created by onishinobuhiro on 2017/09/23.
 */
public class MnistHandler {
    private static final Logger logger = LoggerFactory.getLogger(MnistHandler.class);

    final String TRAIN_IMAGE_FILE = "train-images-idx3-ubyte.gz";
    final String TRAIN_LABEL_FILE = "train-labels-idx1-ubyte.gz";
    final String TEST_IMAGE_FILE = "t10k-images-idx3-ubyte.gz";
    final String TEST_LABEL_FILE = "t10k-labels-idx1-ubyte.gz";
    final String BASE_URL = "http://yann.lecun.com/exdb/mnist/";
    final String BASE_PATH = "./mnist/";

    public MnistHandler() {}

    public MnistData loadMnist(boolean normalize) throws IOException {
        logger.info("----- load MNIST data start.");
        download(TRAIN_IMAGE_FILE);
        download(TRAIN_LABEL_FILE);
        download(TEST_IMAGE_FILE);
        download(TEST_LABEL_FILE);

        double[][] trainData = getFeatures(TRAIN_IMAGE_FILE, normalize);
        double[][] trainLabelList = getLabels(TRAIN_LABEL_FILE);

        double[][] testData = getFeatures(TEST_IMAGE_FILE, normalize);
        double[][] testLabelList = getLabels(TEST_LABEL_FILE);

        MnistData mnistData = new MnistData(trainData, trainLabelList, testData, testLabelList);
        logger.info("----- load MNIST data end.");
        return mnistData;
    }

    public void showImage(String filename, int index) throws IOException {
        BufferedImage image = makeImage(filename, index);
        Icon icon = new ImageIcon(image);
        try {
            JOptionPane.showMessageDialog(null, "index: " + String.valueOf(index), "MnistImageViewer", JOptionPane.PLAIN_MESSAGE, icon);
        } catch (HeadlessException e) {
            logger.error("Can't show image", e);
        }
    }

    private BufferedImage makeImage(String filename, int index) throws IOException {
        BufferedImage image = new BufferedImage(28, 28, BufferedImage.TYPE_INT_RGB);
        double[][] features = getFeatures(filename, true);

        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                int value = (int) (features[index][i * 28 + j] * 255.0);
                image.setRGB(j, i, 0xff000000 | value << 16 | value << 8 | value);
            }
        }
        return image;
    }

    private double[][] getFeatures(String fileName, boolean normalize) throws IOException {
        DataInputStream is = new DataInputStream((new GZIPInputStream(new FileInputStream(BASE_PATH + fileName))));
        // read first 16 bytes.
        is.readInt();
        int numImages = is.readInt();
        int numDimensions = is.readInt() * is.readInt();
        logger.info("numImages: {}", numImages);
        logger.info("numDimensions: {}", numDimensions);

        double[][] features = new double[numImages][numDimensions];
        for (int i = 0; i < numImages; i++) {
            for (int j = 0; j < numDimensions; j++) {
                if (normalize) {
                    features[i][j] = (double) is.readUnsignedByte() / 255.0;
                } else {
                    features[i][j] = (double) is.readUnsignedByte();
                }
            }
        }
        return features;
    }

    private double[][] getLabels(String fileName) throws IOException {
        //ArrayList<Label> labelList = new ArrayList<>();
        DataInputStream is = new DataInputStream(new GZIPInputStream((new FileInputStream(BASE_PATH + fileName))));
        is.readInt();
        int numLabels = is.readInt();
        int numDimensions = 10;

        double[][] labels = new double[numLabels][numDimensions];
        for (int i = 0; i < numLabels; i++) {
            int labelValue = is.readUnsignedByte();
            for (int j = 0; j < numDimensions; j++) {
                labels[i][j] = (j == labelValue) ? 1 : 0;
            }
        }
        return labels;
    }

    //TODO 消す
    /*
    private ArrayList<Label> getLabels(String fileName) throws IOException {
        ArrayList<Label> labelList = new ArrayList<>();
        DataInputStream is = new DataInputStream(new GZIPInputStream((new FileInputStream(BASE_PATH + fileName))));
        is.readInt();
        int numLabels = is.readInt();

        for (int i = 0; i < numLabels; i++) {
            int labelValue = is.readUnsignedByte();
            double[] oneHotLabel = new double[10];
            for (int j = 0; j < 10; j++) {
                oneHotLabel[j] = (j == labelValue) ? 1 : 0;
            }
            Label label = new Label(labelValue, new DoubleMatrix(oneHotLabel).transpose());
            labelList.add(label);
        }
        return labelList;
    }
    */


    private void download(String fileName) throws IOException {
        File baseDir = new File(BASE_PATH);
        if (!baseDir.exists()) {
            baseDir.mkdir();
        }

        if (!new File(BASE_PATH + fileName).exists()) {
            logger.info("Downloading {}{}...", BASE_URL, fileName);
            URL url = new URL(BASE_URL + fileName);
            URLConnection conn = url.openConnection();
            File file = new File(BASE_PATH + fileName);
            InputStream in = conn.getInputStream();
            FileOutputStream out = new FileOutputStream(file, false);
            byte[] data = new byte[1024];
            while (true) {
                int ret = in.read(data);
                if (ret == -1) {
                    break;
                }
                out.write(data, 0, ret);
            }
        } else {
            logger.info("{} is already exists, skip download process...", BASE_PATH + fileName);
        }
    }
}

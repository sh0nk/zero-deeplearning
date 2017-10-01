package com.github.nobby.zerodl.com.github.nobby.zerodl.dataset;

import org.jblas.DoubleMatrix;

/**
 * Created by onishinobuhiro on 2017/10/01.
 * class for Mnist Label file
 */
public class Label {
    private int labelValue;
    private DoubleMatrix label;

    Label(int labelValue, DoubleMatrix label) {
        this.labelValue = labelValue;
        this.label = label;
    }

    public void setLabelIndex(int labelValue) {
        this.labelValue = labelValue;
    }

    public int getLabelValue() {return  this.labelValue;}

    public void setLabel(DoubleMatrix label) {
        this.label = label;
    }

    public DoubleMatrix getLabel() {return  this.label;}
}

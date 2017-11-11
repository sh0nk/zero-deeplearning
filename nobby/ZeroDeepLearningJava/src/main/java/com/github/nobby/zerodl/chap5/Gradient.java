package com.github.nobby.zerodl.chap5;

import lombok.Data;
import org.jblas.DoubleMatrix;

/**
 * Created by onishinobuhiro on 2017/10/07.
 */
@Data
public class Gradient {
    DoubleMatrix W1;
    DoubleMatrix B1;
    DoubleMatrix W2;
    DoubleMatrix B2;
}


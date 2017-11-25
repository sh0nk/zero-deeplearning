package com.github.nobby.zerodl;

import com.github.nobby.zerodl.common.Functions;
import org.jblas.DoubleMatrix;
import org.junit.Test;

public class TempTest {
    @Test
    public void checkMatrix() {
        double[][] hoge = {
                {1,1,7,1,-1},
                {2,-1,2,-2,2},
                {3,10,3,3,-3}
        };
        double[][] fuga = {
                {1,1,7,1,-1}
        };

        DoubleMatrix a = new DoubleMatrix(hoge);
        DoubleMatrix b = new DoubleMatrix(fuga).transpose();
        DoubleMatrix rowMaxs = a.rowMaxs();
        DoubleMatrix columnMaxs = a.columnMaxs();
        Functions.printMatrix(rowMaxs);
        Functions.printMatrix(columnMaxs);
        return;
    }
}

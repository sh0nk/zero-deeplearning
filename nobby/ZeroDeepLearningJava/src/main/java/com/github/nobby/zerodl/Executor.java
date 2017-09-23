package com.github.nobby.zerodl;

import com.github.nobby.zerodl.com.github.nobby.zerodl.chap4.TrainNewralNet;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Profile;

@SpringBootApplication
@Profile("!test")
public class Executor implements CommandLineRunner {


    public static void main(String[] args) {
        SpringApplication.run(Executor.class, args);
    }

    @Override
    public void run(String... arg0) throws Exception {
        TrainNewralNet trainNewralNet = new TrainNewralNet();
        trainNewralNet.exec();
    }
}

package com.example.tempter_predict;

import android.graphics.Bitmap;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.nnapi.NnApiDelegate;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.File;

public class TFLiteUtil {

    private static final String TAG=TFLiteUtil.class.getName();
    private Interpreter tfLite;
    private TensorBuffer inputBuffer;
    private  TensorBuffer outputBuffer;
    private static final int NUM_THREADS=4;
//    float[][][] data = new float[1][120][3];

//    测试数据,shape{1,120,3} {batch_size,step,input_size}
    float[][][] data= {{{996.52f,-8.02f,93.3f},
        {996.57f,-8.41f,93.4f},
        {996.53f,-8.51f,93.9f},
        {996.51f,-8.31f,94.2f},
        {996.51f,-8.27f,94.1f},
        {996.5f,-8.05f,94.4f},
        {996.5f,-7.62f,94.8f},
        {996.5f,-7.62f,94.4f},
        {996.5f,-7.91f,93.8f},
        {996.53f,-8.43f,93.1f},
        {996.62f,-8.76f,93.1f},
        {996.62f,-8.88f,93.2f},
        {996.63f,-8.85f,93.5f},
        {996.74f,-8.83f,93.5f},
        {996.81f,-8.66f,93.9f},
        {996.81f,-8.66f,93.6f},
        {996.86f,-8.7f,93.5f},
        {996.84f,-8.81f,93.5f},
        {996.87f,-8.84f,93.5f},
        {996.97f,-8.94f,93.3f},
        {997.08f,-8.94f,93.4f},
        {997.1f,-8.86f,93.1f},
        {997.06f,-8.99f,92.4f},
        {996.99f,-9.05f,92.6f},
        {997.05f,-9.23f,92.2f},
        {997.11f,-9.49f,92f},
        {997.19f,-9.5f,92.3f},
        {997.24f,-9.35f,92.8f},
        {997.37f,-9.47f,92.4f},
        {997.46f,-9.63f,92.2f},
        {997.43f,-9.67f,92.6f},
        {997.42f,-9.68f,92f},
        {997.53f,-9.9f,91.7f},
        {997.6f,-9.91f,92.4f},
        {997.62f,-9.51f,93.4f},
        {997.71f,-9.67f,92.7f},
        {997.81f,-9.59f,93.2f},
        {997.86f,-9.15f,93.3f},
        {998f,-8.91f,92.5f},
        {998.14f,-9.04f,91.9f},
        {998.21f,-9.43f,91.3f},
        {998.33f,-9.17f,92.9f},
        {998.5f,-8.71f,93f},
        {998.59f,-8.55f,93f},
        {998.79f,-8.4f,93.1f},
        {998.86f,-8.3f,93.1f},
        {999.04f,-8.13f,93.2f},
        {999.17f,-8.1f,92.8f},
        {999.27f,-8.14f,92.6f},
        {999.33f,-8.06f,92.7f},
        {999.44f,-7.95f,92.6f},
        {999.46f,-7.74f,93f},
        {999.59f,-7.57f,92f},
        {999.69f,-7.66f,91.2f},
        {999.79f,-7.71f,91.3f},
        {999.81f,-7.56f,91.7f},
        {999.83f,-7.29f,92.2f},
        {999.96f,-7.15f,92.1f},
        {1000.13f,-7.02f,92.2f},
        {1000.27f,-7.04f,91.6f},
        {1000.43f,-7.03f,91.6f},
        {1000.54f,-7.15f,91.1f},
        {1000.68f,-7.26f,91f},
        {1000.78f,-7.34f,90.8f},
        {1000.83f,-7.35f,90.9f},
        {1000.87f,-7.41f,90.7f},
        {1000.81f,-7.48f,90.5f},
        {1000.74f,-7.38f,90.6f},
        {1000.61f,-7.21f,90.2f},
        {1000.5f,-7.16f,89.8f},
        {1000.36f,-7.03f,89.6f},
        {1000.3f,-6.87f,89.6f},
        {1000.21f,-6.77f,89.5f},
        {1000.18f,-6.7f,89.8f},
        {1000.14f,-6.61f,89.7f},
        {1000.02f,-6.51f,89.5f},
        {1000.02f,-6.21f,89.4f},
        {1000.03f,-5.89f,88.6f},
        {999.97f,-5.83f,87.8f},
        {999.97f,-5.76f,87.7f},
        {1000.02f,-5.9f,87.5f},
        {999.89f,-5.97f,88.5f},
        {999.81f,-5.88f,88.6f},
        {999.81f,-5.94f,89.1f},
        {999.81f,-5.84f,89.6f},
        {999.8f,-5.76f,89.8f},
        {999.81f,-5.75f,89.8f},
        {999.82f,-5.76f,90.2f},
        {999.83f,-5.73f,90.3f},
        {999.88f,-5.69f,90.4f},
        {999.98f,-5.53f,90.2f},
        {1000.06f,-5.57f,89.8f},
        {1000.04f,-5.43f,90f},
        {1000f,-5.32f,89.5f},
        {999.95f,-5.36f,89.2f},
        {999.94f,-5.4f,89.4f},
        {1000.05f,-5.31f,89.9f},
        {1000.05f,-5.28f,89.8f},
        {1000.1f,-5.32f,89.5f},
        {1000.17f,-5.29f,89.7f},
        {1000.13f,-5.33f,89.2f},
        {1000.17f,-5.37f,89.4f},
        {1000.17f,-5.43f,89.3f},
        {1000.18f,-5.28f,89.8f},
        {1000.18f,-5.21f,89.2f},
        {1000.17f,-5.21f,88.9f},
        {1000.16f,-5.24f,88.9f},
        {1000.16f,-5.25f,89.1f},
        {1000.13f,-5.16f,89.1f},
        {1000.07f,-5.12f,89.1f},
        {1000.11f,-5.04f,88.9f},
        {1000.18f,-5.01f,88.7f},
        {1000.23f,-5.12f,88.7f},
        {1000.22f,-5.11f,89.4f},
        {1000.3f,-4.9f,89.4f},
        {1000.19f,-4.86f,88.9f},
        {1000.18f,-4.9f,88.7f},
        {1000.14f,-4.97f,88.7f},
        {1000.18f,-4.99f,89f},
        {1000.22f,-4.9f,89.3f}}};


    /*
     *@param modelPath  model path
     */
    public TFLiteUtil(String modelPath) throws Exception{

        File file=new File(modelPath);

        if(!file.exists()){
            throw new Exception("model file is not exists!");
        }

        try{
            Interpreter.Options options=new Interpreter.Options();
//            使用多线程预测
            options.setNumThreads(NUM_THREADS);
//            使用Android自带的API 或者GPU进行加速
//            NnApiDelegate delegate =new NnApiDelegate();
////            GpuDelegate delegate1=new GpuDelegate();
//            options.addDelegate(delegate);
            tfLite=new Interpreter(file,options);

            //获取图片的输入，shape={1,height,width,3}
            int[] inputShape=tfLite.getInputTensor(tfLite.getInputIndex("serving_default_x:0")).shape();
            DataType inputDataType=tfLite.getInputTensor(tfLite.getInputIndex("serving_default_x:0")).dataType();
            inputBuffer= TensorBuffer.createFixedSize(inputShape, inputDataType);

//            TensorBuffer tmpBuffer = TensorBuffer.createDynamic(DataType.FLOAT32);
//
//            tmpBuffer.loadArray(data);
//            inputBuffer.loadBuffer(tmpBuffer.getBuffer());

            int[] outputShape=tfLite.getOutputTensor(tfLite.getOutputIndex("StatefulPartitionedCall:0")).shape();
            DataType outputDataType=tfLite.getOutputTensor(tfLite.getOutputIndex("StatefulPartitionedCall:0")).dataType();
            outputBuffer = TensorBuffer.createFixedSize(outputShape, outputDataType);

        } catch (Exception e) {
            e.printStackTrace();
            throw new Exception("load model fail!");
        }
    }

    public float[] predict() throws  Exception{
        try{
            tfLite.run(data,outputBuffer.getBuffer().rewind());
//            tfLite.run(inputBuffer.getBuffer(),outputBuffer.getBuffer().rewind());
        }catch (Exception e){
            throw new Exception("predict image fail! log:" + e);
        }

        float[] results=outputBuffer.getFloatArray();

        return results;
    }


}

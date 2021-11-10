package com.example.tempter_predict;

import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import java.io.File;

public class MainActivity extends AppCompatActivity {

    private TFLiteUtil tfUtil;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

//        Android 6 以上的设备还要动态申请权限
        if (!hasPermission()) {
            requestPermission();
        }

//        我们是把模型放在Android项目的assets目录的，但是Tensorflow Lite并不建议直接在assets读取模型，
//        所以我们需要把模型复制到一个缓存目录，然后再从缓存目录加载模型，同时还有读取标签名，
//        标签名称按照训练的label顺序存放在assets的label_list.txt，
        String classificationModelPath = getCacheDir().getAbsolutePath() + File.separator + "multi_tempter_back.tflite";
        Utils.copyFileFromAsset(MainActivity.this, "multi_tempter_back.tflite", classificationModelPath);
        try {
            tfUtil = new TFLiteUtil(classificationModelPath);
            Toast.makeText(MainActivity.this, "模型加载成功！", Toast.LENGTH_SHORT).show();
        } catch (Exception e) {
            Toast.makeText(MainActivity.this, "模型加载失败！", Toast.LENGTH_SHORT).show();
            e.printStackTrace();
            finish();
        }

        // 获取控件
        Button res_btn = findViewById(R.id.res_btn);
        // 获取控件
        TextView txt=(TextView)findViewById(R.id.txt_id);


        String TAG="results";

        res_btn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                try {
                    String res="预测结果为：";
                    float[] results= tfUtil.predict();
                    for (int i=0;i<results.length;i++){
                        Log.i(TAG, res+results[i] );
                        res=res+String.valueOf(results[i])+" ";
                    }
                    txt.setText(res);

                } catch (Exception e) {
                    e.printStackTrace();
                }


            }
        });

    }

    // check had permission
    private boolean hasPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            return
                    checkSelfPermission(Manifest.permission.READ_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED &&
                    checkSelfPermission(Manifest.permission.WRITE_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED;
        } else {
            return true;
        }
    }

    // request permission
    private void requestPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            requestPermissions(new String[]{
                    Manifest.permission.READ_EXTERNAL_STORAGE,
                    Manifest.permission.WRITE_EXTERNAL_STORAGE}, 1);
        }
    }
}
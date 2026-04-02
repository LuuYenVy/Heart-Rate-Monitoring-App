package com.example.heartdiseaseprediction.Activities.UI.Splash;

import android.content.Intent;
import android.os.Bundle;
import android.os.Handler;

import androidx.appcompat.app.AppCompatActivity;
import androidx.lifecycle.ViewModelProvider;

public class SplashActivity extends AppCompatActivity {

    private static final int SPLASH_TIME_OUT = 2000;
    SplashViewModel viewModel;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        viewModel = new ViewModelProvider(this).get(SplashViewModel.class);

        new Handler().postDelayed(() -> {
            viewModel.checkUser(this);
        }, SPLASH_TIME_OUT);

        viewModel.getNavigateTo().observe(this, destination -> {
            startActivity(new Intent(this, destination));
            finish();
        });
    }
}

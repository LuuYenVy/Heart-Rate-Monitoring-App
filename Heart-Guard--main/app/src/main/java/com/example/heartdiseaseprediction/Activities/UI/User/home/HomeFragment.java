package com.example.heartdiseaseprediction.Activities.UI.User.home;

import androidx.fragment.app.Fragment;
import android.graphics.Color;
import android.media.MediaPlayer;
import android.os.Bundle;

import androidx.annotation.NonNull;
import androidx.fragment.app.Fragment;
import androidx.lifecycle.ViewModel;
import androidx.lifecycle.ViewModelProvider;

import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.ImageButton;
import android.widget.TextView;
import android.widget.Toast;

import com.example.heartdiseaseprediction.R;
import com.mikhaellopez.circularprogressbar.CircularProgressBar;


public class HomeFragment extends Fragment {
    private TextView tvHeartRate, tvTime, tvPredict;
    private CircularProgressBar progressBar;
    private HomeViewModel viewModel;
    long k = 1000;

    public View onCreateView(LayoutInflater inflater, ViewGroup container, Bundle saveInstanceState) {
        View view = inflater.inflate(R.layout.fragment_home, container, false);
        tvHeartRate = view.findViewById(R.id.textView_HeartRate);
        tvPredict = view.findViewById(R.id.textView_PredictResult);
        progressBar = view.findViewById(R.id.circularProgressBar);
        tvTime= view.findViewById(R.id.lastResultDate);
        viewModel = new ViewModelProvider(this).get(HomeViewModel.class);
        viewModel.pushFakeData();
        observeData();
        viewModel.startListening();
        return view;
    }

    private void observeData() {
        viewModel.getHeartData().observe(getViewLifecycleOwner(), result -> {

            tvHeartRate.setText(result.getBeat());
            tvTime.setText(result.getTime());

            int progress = (int) (Float.parseFloat(result.getBeat()) / 250 * 100);
            progressBar.setProgressWithAnimation(progress, k);

            if (result.isDanger()) {
                tvPredict.setTextColor(Color.RED);
                tvPredict.setText("Danger!!!");

                MediaPlayer mp = MediaPlayer.create(getContext(), R.raw.beep_warning);
                mp.start();

            } else {
                tvPredict.setTextColor(Color.BLACK);
                tvPredict.setText("Normal");
            }
        });

    }
}

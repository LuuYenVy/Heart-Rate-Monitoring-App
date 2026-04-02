package com.example.heartdiseaseprediction.Activities.UI.User.appointment;

import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.ImageView;
import android.widget.Spinner;
import androidx.lifecycle.ViewModelProvider;

import androidx.appcompat.app.AppCompatActivity;

import com.example.heartdiseaseprediction.Activities.UI.User.MainActivity;
import com.example.heartdiseaseprediction.Activities.UI.User.schedule.ScheduleActivity;
import com.example.heartdiseaseprediction.R;

import java.util.List;
public class ChooseServiceActivity extends AppCompatActivity {

    private AppointmentViewModel viewModel;
    private Spinner spinner;
    private ImageView schedule, Back_btn;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_chooce_service);

        viewModel = new ViewModelProvider(this).get(AppointmentViewModel.class);

        initView();
        observeData();
        viewModel.loadServices();
    }

    private void initView() {
        spinner = findViewById(R.id.ChooseService);
        schedule = findViewById(R.id.schedule);
        Back_btn = findViewById(R.id.ButtonBack);

        schedule.setOnClickListener(v -> {
            startActivity(new Intent(this, ScheduleActivity.class));
            finish();
        });

        Back_btn.setOnClickListener(v -> {
            startActivity(new Intent(this, MainActivity.class));
        });
    }

    private void setupSpinner(List<String> services) {

        ArrayAdapter<String> adapter = new ArrayAdapter<>(
                this,
                android.R.layout.simple_spinner_item,
                services
        );

        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        spinner.setAdapter(adapter);

        spinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parentView, View selectedItemView, int position, long id) {
                String service = (String) parentView.getItemAtPosition(position);
                viewModel.selectService(service);
            }

            @Override
            public void onNothingSelected(AdapterView<?> parentView) {}
        });
    }

    private void observeData() {
        viewModel.getServices().observe(this, list -> {
            if (list != null) {
                setupSpinner(list);
            }
        });

        viewModel.getSelectedService().observe(this, service -> {
            if (service != null) {
                saveToPrefs(service);
            }
        });
    }

    private void saveToPrefs(String service) {
        SharedPreferences sharedPreferences = getSharedPreferences("MyPrefs", Context.MODE_PRIVATE);
        SharedPreferences.Editor editor = sharedPreferences.edit();
        editor.putString("selectedService", service);
        editor.apply();
    }
}

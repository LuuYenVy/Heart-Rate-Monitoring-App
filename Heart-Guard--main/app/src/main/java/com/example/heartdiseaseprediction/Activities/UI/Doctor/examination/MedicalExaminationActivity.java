package com.example.heartdiseaseprediction.Activities.UI.Doctor.examination;


import android.content.Intent;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
import androidx.lifecycle.ViewModelProvider;

import com.example.heartdiseaseprediction.Activities.UI.Doctor.prescription.PrescriptionsActivity;
import com.example.heartdiseaseprediction.R;
import com.google.android.material.textfield.TextInputEditText;

public class MedicalExaminationActivity extends AppCompatActivity {

    TextView userInfo, weight, height, service, date;
    EditText symptom, medicalhistory;
    TextInputEditText diagnostic;
    ImageView prescriptions;

    ExaminationViewModel viewModel;

    String appointmentId;
    String temperature;
    String bloodPressure;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_medical_examination);

        initView();

        viewModel = new ViewModelProvider(this).get(ExaminationViewModel.class);

        getData();
        observeData();
        setupClick();

        viewModel.loadAppointment(appointmentId);
    }

    private void initView() {
        userInfo = findViewById(R.id.userInfo);
        weight = findViewById(R.id.weight);
        height = findViewById(R.id.height);
        service = findViewById(R.id.service);
        date = findViewById(R.id.date);
        symptom = findViewById(R.id.symptom);
        medicalhistory = findViewById(R.id.medicalHistory);
        diagnostic = findViewById(R.id.diagnostic);
        prescriptions = findViewById(R.id.prescriptions);
    }

    private void getData() {

        appointmentId = getIntent().getStringExtra("APPOINTMENT_ID");

        if (appointmentId == null) {
            Toast.makeText(this, "Appointment ID null", Toast.LENGTH_SHORT).show();
            finish();
            return;
        }

        SharedPreferences pref = getSharedPreferences("MyPrefs", MODE_PRIVATE);
        temperature = pref.getString("temperater", "");
        bloodPressure = pref.getString("blood_pressure", "");
    }

    private void observeData() {

        viewModel.getAppointment().observe(this, ap -> {
            if (ap != null) {
                userInfo.setText(ap.getUser().getUsername());
                weight.setText(ap.getUser().getWeight());
                height.setText(ap.getUser().getHeight());
                service.setText(ap.getService());
                date.setText(ap.getDate());
            }
        });

        viewModel.getSuccess().observe(this, isSuccess -> {
            if (isSuccess != null && isSuccess) {
                Toast.makeText(this, "Update success", Toast.LENGTH_SHORT).show();
                sendToPrescription();
            }
        });

        viewModel.getError().observe(this, err -> {
            Toast.makeText(this, err, Toast.LENGTH_SHORT).show();
        });
    }

    private void setupClick() {

        prescriptions.setOnClickListener(v -> {

            String s = symptom.getText().toString();
            String h = medicalhistory.getText().toString();
            String d = diagnostic.getText().toString();

            if (viewModel.validate(s, h, d)) {
                viewModel.updateAppointment(
                        appointmentId,
                        temperature,
                        bloodPressure,
                        s,
                        h,
                        d
                );
            }
        });
    }

    private void sendToPrescription() {
        Intent intent = new Intent(this, PrescriptionsActivity.class);

        intent.putExtra("APPOINTMENT_ID", appointmentId);

        startActivity(intent);
        finish();
    }
}
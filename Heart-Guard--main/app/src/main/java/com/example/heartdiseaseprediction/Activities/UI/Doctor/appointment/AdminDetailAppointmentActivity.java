package com.example.heartdiseaseprediction.Activities.UI.Doctor.appointment;

import android.content.Intent;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
import androidx.lifecycle.ViewModelProvider;

import com.example.heartdiseaseprediction.Activities.UI.Doctor.main.AdminMainActivity;
import com.example.heartdiseaseprediction.Activities.UI.Doctor.receipt.AdminReceptActivity;
import com.example.heartdiseaseprediction.R;
import com.google.android.material.textfield.TextInputEditText;

public class AdminDetailAppointmentActivity extends AppCompatActivity {

    TextView weight, height, userInfo, servicename, date, symptom, medicalhistory;
    TextInputEditText diagnostic;
    ImageView Back_btn;
    LinearLayout Prescriptions;

    DoctorAppointmentViewModel viewModel;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_detail_appointment_admin);

        initView();

        viewModel = new ViewModelProvider(this).get(DoctorAppointmentViewModel.class);

        SharedPreferences pref = getSharedPreferences("MyPrefs", MODE_PRIVATE);
        String appointmentId = pref.getString("AppointmentKey", "");

        observeData();

        viewModel.loadAppointmentDetail(appointmentId);

        setupClick();
    }

    private void initView() {
        userInfo = findViewById(R.id.userInfo);
        weight = findViewById(R.id.weight);
        height = findViewById(R.id.height);
        servicename = findViewById(R.id.service);
        date = findViewById(R.id.date);
        symptom = findViewById(R.id.symptom);
        medicalhistory = findViewById(R.id.medicalHistory);
        diagnostic = findViewById(R.id.diagnostic);
        Back_btn = findViewById(R.id.ButtonBack);
        Prescriptions = findViewById(R.id.Recept);
    }

    private void observeData() {

        // Appointment
        viewModel.getAppointment().observe(this, app -> {
            if (app != null) {
                servicename.setText(app.getService());
                date.setText(app.getDate());
            }
        });

        // User
        viewModel.getUser().observe(this, user -> {
            if (user != null) {
                weight.setText(user.getWeight());
                height.setText(user.getHeight());
                userInfo.setText(user.getUsername() + " - " + user.getAge() + " - " + user.getGender());
            }
        });

        viewModel.getSymptom().observe(this, s -> symptom.setText(s));
        viewModel.getMedicalHistory().observe(this, m -> medicalhistory.setText(m));
        viewModel.getDiagnostic().observe(this, d -> diagnostic.setText(d));

        viewModel.getDoctorName().observe(this, name -> {
            getSharedPreferences("DoctorPrefs", MODE_PRIVATE)
                    .edit()
                    .putString("DoctorName", name)
                    .apply();
        });

        viewModel.getError().observe(this, err -> {
            Toast.makeText(this, err, Toast.LENGTH_SHORT).show();
        });
    }

    private void setupClick() {

        Back_btn.setOnClickListener(v -> {
            startActivity(new Intent(this, AdminMainActivity .class));
        });

        Prescriptions.setOnClickListener(v -> {
            startActivity(new Intent(this, AdminReceptActivity.class));
            finish();
        });
    }
}

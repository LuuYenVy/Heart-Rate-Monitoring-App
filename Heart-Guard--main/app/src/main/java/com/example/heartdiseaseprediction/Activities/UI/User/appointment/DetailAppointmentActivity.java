package com.example.heartdiseaseprediction.Activities.UI.User.appointment;

import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
import androidx.lifecycle.ViewModelProvider;

import com.example.heartdiseaseprediction.Activities.UI.User.MainActivity;
import com.example.heartdiseaseprediction.Activities.UI.User.appointment.AppointmentViewModel;
import com.example.heartdiseaseprediction.Activities.UI.User.receipt.ReceiptActivity;
import com.example.heartdiseaseprediction.Activities.data.Model.Appointment;
import com.example.heartdiseaseprediction.Activities.data.Model.User;
import com.example.heartdiseaseprediction.R;
import com.example.heartdiseaseprediction.SesionManager.AppointmentSessionManager;
import com.example.heartdiseaseprediction.SesionManager.UserSessionManager;
import com.google.android.material.textfield.TextInputEditText;

public class DetailAppointmentActivity extends AppCompatActivity {

    LinearLayout prescriptions;
    TextView weight, height, userInfo, servicename, date, symptom, medicalhistory;
    TextInputEditText diagnostic;
    ImageView backBtn;

    AppointmentSessionManager appointmentSessionManager;
    UserSessionManager userSessionManager;

    AppointmentViewModel viewModel;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_detail_apointment);

        initUI();
        initViewModel();
        loadBaseInfo();
        loadAppointmentDetail();
    }

    private void initUI() {
        userInfo = findViewById(R.id.userInfo);
        weight = findViewById(R.id.weight);
        height = findViewById(R.id.height);
        servicename = findViewById(R.id.service);
        date = findViewById(R.id.date);
        symptom = findViewById(R.id.symptom);
        medicalhistory = findViewById(R.id.medicalHistory);
        diagnostic = findViewById(R.id.diagnostic);

        prescriptions = findViewById(R.id.Recept);
        backBtn = findViewById(R.id.ButtonBack);

        prescriptions.setOnClickListener(v -> {
            Intent intent = new Intent(this, ReceiptActivity.class);
            startActivity(intent);
            finish();
        });

        backBtn.setOnClickListener(v -> {
            startActivity(new Intent(this, MainActivity.class));
        });

        userSessionManager = new UserSessionManager(getApplicationContext());
        appointmentSessionManager = new AppointmentSessionManager(getApplicationContext());
    }

    private void initViewModel() {
        viewModel = new ViewModelProvider(this).get(AppointmentViewModel.class);

        viewModel.getAppointmentLiveData().observe(this, appointment -> {
            if (appointment != null) {
                bindDetail(appointment);
            }
        });

        viewModel.getErrorLiveData().observe(this, error -> {
            Toast.makeText(this, error, Toast.LENGTH_SHORT).show();
        });
    }

    private void loadBaseInfo() {
        User user = userSessionManager.getUserDetails();

        SharedPreferences prefs = getSharedPreferences("MyPrefs", Context.MODE_PRIVATE);
        String service = prefs.getString("NameApoimentSelect", "");
        String dateStr = prefs.getString("DateApoimentSelect", "");

        if (user != null) {
            weight.setText(user.getWeight());
            height.setText(user.getHeight());
            userInfo.setText(user.getUsername() + " - " + user.getAge() + " - " + user.getGender());
        }

        servicename.setText(service);
        date.setText(dateStr);
    }

    private void loadAppointmentDetail() {
        SharedPreferences prefs = getSharedPreferences("MyPrefs", Context.MODE_PRIVATE);
        String id = prefs.getString("AppointmentKey", "");

        if (id == null || id.isEmpty()) {
            Toast.makeText(this, "Appointment ID not found", Toast.LENGTH_SHORT).show();
            return;
        }

        viewModel.loadAppointment(id);
    }

    private void bindDetail(Appointment ap) {
        symptom.setText(ap.getSypptom());
        medicalhistory.setText(ap.getMedical_history());
        diagnostic.setText(ap.getDiagnostic());

        SharedPreferences.Editor editor =
                getSharedPreferences("DoctorPrefs", MODE_PRIVATE).edit();

        editor.putString("DoctorName", ap.getDoctor());
        editor.apply();
    }
}

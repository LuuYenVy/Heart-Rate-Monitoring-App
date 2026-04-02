package com.example.heartdiseaseprediction.Activities.UI.User.schedule;

import android.app.DatePickerDialog;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.widget.*;

import androidx.appcompat.app.AppCompatActivity;
import androidx.lifecycle.ViewModelProvider;

import com.example.heartdiseaseprediction.Activities.UI.User.MainActivity;
import com.example.heartdiseaseprediction.Activities.data.Model.Appointment;
import com.example.heartdiseaseprediction.Activities.data.Model.User;
import com.example.heartdiseaseprediction.R;
import com.example.heartdiseaseprediction.SesionManager.UserSessionManager;
import com.google.android.material.textfield.TextInputEditText;
import com.google.firebase.auth.FirebaseAuth;

import java.util.Calendar;

public class ScheduleActivity extends AppCompatActivity {

    private ScheduleViewModel viewModel;

    private TextView tvDate;
    private Spinner spinnerDoctor;
    private TextInputEditText edtNote;

    private int year, month, day;
    private String selectedService;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_schedule);

        viewModel = new ViewModelProvider(this).get(ScheduleViewModel.class);

        tvDate = findViewById(R.id.date);
        spinnerDoctor = findViewById(R.id.Doctor);
        edtNote = findViewById(R.id.note);

        ImageView backBtn = findViewById(R.id.ButtonBack);
        Button btnMake = findViewById(R.id.makeApointment);

        // get service
        SharedPreferences prefs = getSharedPreferences("MyPrefs", Context.MODE_PRIVATE);
        selectedService = prefs.getString("selectedService", "");

        observeData();

        viewModel.loadDoctors(selectedService);

        // date picker
        findViewById(R.id.imageButton2).setOnClickListener(v -> showDatePicker());

        // create appointment
        btnMake.setOnClickListener(v -> createAppointment());

        backBtn.setOnClickListener(v -> {
            startActivity(new Intent(this, MainActivity.class));
        });
    }

    private void observeData() {

        viewModel.getDoctorList().observe(this, list -> {
            ArrayAdapter<String> adapter = new ArrayAdapter<>(this,
                    android.R.layout.simple_spinner_item, list);
            adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
            spinnerDoctor.setAdapter(adapter);
        });

        viewModel.getCreateSuccess().observe(this, success -> {
            if (success) {
                Toast.makeText(this, "Schedule Successfully!", Toast.LENGTH_SHORT).show();
                startActivity(new Intent(this, MainActivity.class));
            }
        });

        viewModel.getError().observe(this,
                err -> Toast.makeText(this, err, Toast.LENGTH_SHORT).show());
    }

    private void showDatePicker() {
        Calendar calendar = Calendar.getInstance();

        year = calendar.get(Calendar.YEAR);
        month = calendar.get(Calendar.MONTH);
        day = calendar.get(Calendar.DAY_OF_MONTH);

        new DatePickerDialog(this, (view, y, m, d) -> {
            year = y;
            month = m;
            day = d;

            tvDate.setText(d + "/" + (m + 1) + "/" + y);

        }, year, month, day).show();
    }

    private void createAppointment() {

        String date = tvDate.getText().toString();
        String doctor = spinnerDoctor.getSelectedItem().toString();
        String note = edtNote.getText().toString();

        FirebaseAuth auth = FirebaseAuth.getInstance();
        String uid = auth.getCurrentUser().getUid();

        UserSessionManager session = new UserSessionManager(this);
        User user = session.getUserDetails();

        String id = com.google.firebase.database.FirebaseDatabase.getInstance()
                .getReference("appointments")
                .push().getKey();

        String userid_status = uid + "_incoming";

        Appointment appointment = new Appointment(
                date,
                doctor,
                note,
                "incoming",
                selectedService,
                user,
                userid_status
        );

        viewModel.createAppointment(id, appointment);
    }
}
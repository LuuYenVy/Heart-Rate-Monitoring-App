package com.example.heartdiseaseprediction.Activities.UI.Doctor.examination;



import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
import androidx.lifecycle.ViewModelProvider;

import com.example.heartdiseaseprediction.Activities.UI.Doctor.main.AdminMainActivity;
import com.example.heartdiseaseprediction.R;
public class PreCheckUpProcessActivity extends AppCompatActivity {

    TextView temperater, blood_pressure;
    ImageView MedicalExamination, Back_btn;

    ExaminationViewModel viewModel;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_precheckup_process);

        temperater = findViewById(R.id.temperater);
        blood_pressure = findViewById(R.id.bloodPressure);
        MedicalExamination = findViewById(R.id.MedicalExamination);
        Back_btn = findViewById(R.id.ButtonBack);

        viewModel = new ViewModelProvider(this).get(ExaminationViewModel.class);

        Back_btn.setOnClickListener(v -> {
            startActivity(new Intent(this, AdminMainActivity.class));
        });

        MedicalExamination.setOnClickListener(v -> {
            String temp = temperater.getText().toString();
            String bp = blood_pressure.getText().toString();

            viewModel.validate(temp, bp);
        });

        viewModel.getIsValid().observe(this, isValid -> {
            if (isValid != null && isValid) {

                viewModel.setPreCheckData(
                        temperater.getText().toString(),
                        blood_pressure.getText().toString()
                );
                Intent intent;
                intent = new Intent(this, MedicalExaminationActivity.class);
                String appointmentId = getIntent().getStringExtra("APPOINTMENT_ID");
                intent.putExtra("APPOINTMENT_ID", appointmentId);
                startActivity(intent);

            } else {
                Toast.makeText(this, "Please fill all fields", Toast.LENGTH_SHORT).show();
            }
        });
    }
}

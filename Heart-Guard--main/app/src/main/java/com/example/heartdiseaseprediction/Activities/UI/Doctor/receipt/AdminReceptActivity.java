package com.example.heartdiseaseprediction.Activities.UI.Doctor.receipt;

import android.content.Intent;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
import androidx.lifecycle.ViewModelProvider;

import com.example.heartdiseaseprediction.Activities.UI.Doctor.appointment.AdminDetailAppointmentActivity;
import com.example.heartdiseaseprediction.R;

import java.util.List;

public class AdminReceptActivity extends AppCompatActivity {

    TextView doctorName;
    ImageView Back_btn;
    LinearLayout container;

    AdminReceptViewModel viewModel;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_recept);

        doctorName = findViewById(R.id.Doctorname);
        Back_btn = findViewById(R.id.ButtonBack);
        container = findViewById(R.id.container_drug);

        viewModel = new ViewModelProvider(this).get(AdminReceptViewModel.class);

        setupDoctorName();
        observeData();
        loadData();
        setupClick();
    }

    private void setupDoctorName() {
        SharedPreferences pref = getSharedPreferences("DoctorPrefs", MODE_PRIVATE);
        doctorName.setText(pref.getString("DoctorName", "Doctor"));
    }

    private void loadData() {
        SharedPreferences pref = getSharedPreferences("MyPrefs", MODE_PRIVATE);
        String appointmentId = pref.getString("AppointmentKey", "");

        viewModel.loadMedicines(appointmentId);
    }

    private void observeData() {

        viewModel.getMedicineList().observe(this, list -> {
            if (list != null) {
                renderMedicines(list);
            }
        });

        viewModel.getError().observe(this, err -> {
            Toast.makeText(this, err, Toast.LENGTH_SHORT).show();
        });
    }

    private void renderMedicines(List<com.example.heartdiseaseprediction.Activities.data.Model.Medicine> list) {

        container.removeAllViews();

        for (com.example.heartdiseaseprediction.Activities.data.Model.Medicine m : list) {

            LayoutInflater inflater = LayoutInflater.from(this);
            android.view.View itemView = inflater.inflate(R.layout.item_receipt, container, false);

            TextView name = itemView.findViewById(R.id.drugname);
            TextView routine = itemView.findViewById(R.id.routine);
            TextView amount = itemView.findViewById(R.id.amount);

            name.setText(m.getName());
            routine.setText(m.getRoutine());
            amount.setText(m.getAmount());

            container.addView(itemView);
        }
    }

    private void setupClick() {
        Back_btn.setOnClickListener(v -> {
            startActivity(new Intent(this, AdminDetailAppointmentActivity.class));
        });
    }
}
package com.example.heartdiseaseprediction.Activities.UI.User.receipt;

import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
import androidx.lifecycle.ViewModelProvider;

import com.example.heartdiseaseprediction.Activities.data.Model.Medicine;
import com.example.heartdiseaseprediction.R;
import com.example.heartdiseaseprediction.Activities.UI.User.appointment.DetailAppointmentActivity;

import java.util.List;

public class ReceiptActivity extends AppCompatActivity {

    private ReceiptViewModel viewModel;

    private LinearLayout container;
    private TextView doctorName;
    private ImageView backBtn;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_recept);

        viewModel = new ViewModelProvider(this).get(ReceiptViewModel.class);

        doctorName = findViewById(R.id.Doctorname);
        backBtn = findViewById(R.id.ButtonBack);

        ScrollView scrollView = findViewById(R.id.scrollViewdrug);
        container = new LinearLayout(this);
        container.setOrientation(LinearLayout.VERTICAL);
        scrollView.addView(container);

        // Doctor name
        SharedPreferences doctorPrefs = getSharedPreferences("DoctorPrefs", MODE_PRIVATE);
        doctorName.setText(doctorPrefs.getString("DoctorName", "Doctor"));

        // Appointment ID
        SharedPreferences prefs = getSharedPreferences("MyPrefs", Context.MODE_PRIVATE);
        String appointmentId = prefs.getString("AppointmentKey", "");

        observeData();

        viewModel.loadMedicines(appointmentId);

        backBtn.setOnClickListener(v -> {
            startActivity(new Intent(this, DetailAppointmentActivity.class));
        });
    }

    private void observeData() {

        viewModel.getMedicineLiveData().observe(this, this::renderMedicines);

        viewModel.getErrorLiveData().observe(this,
                error -> Toast.makeText(this, error, Toast.LENGTH_SHORT).show());
    }

    private void renderMedicines(List<Medicine> list) {
        container.removeAllViews();

        for (Medicine med : list) {
            View itemView = LayoutInflater.from(this)
                    .inflate(R.layout.item_receipt, container, false);

            TextView name = itemView.findViewById(R.id.drugname);
            TextView routine = itemView.findViewById(R.id.routine);
            TextView amount = itemView.findViewById(R.id.amount);

            name.setText(med.getName());
            routine.setText(med.getRoutine());
            amount.setText(med.getAmount());

            container.addView(itemView);
        }
    }
}
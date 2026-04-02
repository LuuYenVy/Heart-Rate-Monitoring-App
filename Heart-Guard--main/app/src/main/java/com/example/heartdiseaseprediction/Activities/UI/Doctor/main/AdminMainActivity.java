package com.example.heartdiseaseprediction.Activities.UI.Doctor.main;

import static android.webkit.ConsoleMessage.MessageLevel.LOG;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;
import com.example.heartdiseaseprediction.Activities.UI.Auth.LoginActivity;
import com.example.heartdiseaseprediction.Activities.UI.Doctor.appointment.AdminDetailAppointmentActivity;
import com.example.heartdiseaseprediction.Activities.UI.Doctor.chart.ChartActivity;
import com.example.heartdiseaseprediction.Activities.UI.Doctor.examination.PreCheckUpProcessActivity;
import com.example.heartdiseaseprediction.Activities.UI.Doctor.prescription.PrescriptionsActivity;
import com.example.heartdiseaseprediction.Activities.data.Model.Appointment;
import com.example.heartdiseaseprediction.R;
import com.example.heartdiseaseprediction.SesionManager.UserSessionManager;
import com.google.firebase.auth.FirebaseAuth;
import androidx.lifecycle.ViewModelProvider;
public class AdminMainActivity extends AppCompatActivity {

    private DoctorViewModel viewModel;
    private LinearLayout incomingLayout, historyLayout;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_admin_main_sceen);
        UserSessionManager session = new UserSessionManager(this);

        String doctorName = session.getDoctorName();

        viewModel = new ViewModelProvider(this).get(DoctorViewModel.class);

        findViewById(R.id.LogOutBtn).setOnClickListener(v -> {
            FirebaseAuth.getInstance().signOut();
            startActivity(new Intent(this, LoginActivity.class));
            finish();
        });

        setupLayout();
        observeData();

        viewModel.loadData(doctorName);
    }

    private void setupLayout() {
        ScrollView scrollIncoming = findViewById(R.id.scrollItemIncoming);
        incomingLayout = new LinearLayout(this);
        incomingLayout.setOrientation(LinearLayout.VERTICAL);
        scrollIncoming.addView(incomingLayout);

        ScrollView scrollHistory = findViewById(R.id.scrollItemHistory);
        historyLayout = new LinearLayout(this);
        historyLayout.setOrientation(LinearLayout.VERTICAL);
        scrollHistory.addView(historyLayout);
    }

    private void observeData() {

        viewModel.getIncoming().observe(this, list -> {
            incomingLayout.removeAllViews();
            for (Appointment ap : list) {
                addItem(ap, true);
            }
        });

        viewModel.getHistory().observe(this, list -> {
            historyLayout.removeAllViews();
            for (Appointment ap : list) {
                addItem(ap, false);
            }
        });

        viewModel.getError().observe(this, err ->
                Toast.makeText(this, err, Toast.LENGTH_SHORT).show()
        );
    }

    private void addItem(Appointment ap, boolean isIncoming) {

        View itemView = LayoutInflater.from(this).inflate(R.layout.item_user, null);

        TextView name = itemView.findViewById(R.id.name);
        TextView username = itemView.findViewById(R.id.username);
        TextView date = itemView.findViewById(R.id.date);

        name.setText(ap.getService());
        username.setText(ap.getUser().getUsername());
        date.setText(ap.getDate());

        itemView.findViewById(R.id.show_more_chart).setOnClickListener(v -> {
            Intent intent = new Intent(this, ChartActivity.class);
            intent.putExtra("USER_ID", ap.getUser().getUserID());
            startActivity(intent);
        });

        itemView.findViewById(R.id.appointmentInfo).setOnClickListener(v -> {

            Intent intent;

            if (isIncoming) {
                intent = new Intent(this, PreCheckUpProcessActivity.class);
            } else {
                intent = new Intent(this, AdminDetailAppointmentActivity.class);
            }


            intent.putExtra("APPOINTMENT_ID", ap.getId());

            intent.putExtra("DOCTOR", ap.getDoctor());

            startActivity(intent);
        });

        if (isIncoming) {
            incomingLayout.addView(itemView);
        } else {
            historyLayout.addView(itemView);
        }
    }
}

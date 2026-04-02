package com.example.heartdiseaseprediction.Activities.UI.Doctor.prescription;
import android.content.Intent;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
import androidx.lifecycle.ViewModelProvider;

import com.example.heartdiseaseprediction.Activities.UI.Doctor.main.AdminMainActivity;
import com.example.heartdiseaseprediction.Activities.data.Model.Medicine;
import com.example.heartdiseaseprediction.R;

import java.util.ArrayList;
import java.util.List;
public class PrescriptionsActivity extends AppCompatActivity {

    ImageView addDrug, Back_btn;
    Button Finish;
    TextView doctorname;

    LinearLayout container;

    PrescriptionViewModel viewModel;

    List<View> itemViews = new ArrayList<>();
    List<String> medicineList = new ArrayList<>();

    String appointmentId;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_prescriptions);

        viewModel = new ViewModelProvider(this).get(PrescriptionViewModel.class);

        container = findViewById(R.id.layoutContent);
        addDrug = findViewById(R.id.addDrug);
        Finish = findViewById(R.id.finish);
        Back_btn = findViewById(R.id.ButtonBack);
        doctorname = findViewById(R.id.Doctorname);

        appointmentId = getIntent().getStringExtra("APPOINTMENT_ID");
        String doctorName = getIntent().getStringExtra("DOCTOR");
        doctorname.setText(doctorName);

        viewModel.loadMedicines();

        viewModel.getMedicineList().observe(this, list -> {
            medicineList = list;
        });

        viewModel.getIsSaved().observe(this, saved -> {
            if (saved) {
                Toast.makeText(this, "Successfully!!", Toast.LENGTH_SHORT).show();
                startActivity(new Intent(this, AdminMainActivity.class));
                finish();
            }
        });

        Back_btn.setOnClickListener(v -> finish());

        addDrug.setOnClickListener(v -> addDrugItem());

        Finish.setOnClickListener(v -> saveData());
    }

    private void addDrugItem() {
        LayoutInflater inflater = LayoutInflater.from(this);
        View itemView = inflater.inflate(R.layout.item_drug, null);

        Spinner spinner = itemView.findViewById(R.id.drugname);

        ArrayAdapter<String> adapter = new ArrayAdapter<>(
                this,
                android.R.layout.simple_spinner_item,
                medicineList
        );
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        spinner.setAdapter(adapter);

        ImageView remove = itemView.findViewById(R.id.removedrug);
        remove.setOnClickListener(v -> {
            container.removeView(itemView);
            itemViews.remove(itemView);
        });

        itemViews.add(itemView);
        container.addView(itemView);
    }

    private void saveData() {

        List<Medicine> list = new ArrayList<>();

        for (View item : itemViews) {

            Spinner drugNameSpinner = item.findViewById(R.id.drugname);
            EditText routineEditText = item.findViewById(R.id.routine);
            EditText amountEditText = item.findViewById(R.id.amount);

            String name = drugNameSpinner.getSelectedItem().toString();
            String routine = routineEditText.getText().toString();
            String amount = amountEditText.getText().toString();

            list.add(new Medicine(name, routine, amount));
        }

        viewModel.saveMedicines(appointmentId, list);
    }
}

package com.example.heartdiseaseprediction.Activities.UI.Doctor.chart;

import android.app.DatePickerDialog;
import android.content.Intent;
import android.os.Bundle;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
import androidx.lifecycle.ViewModelProvider;

import com.example.heartdiseaseprediction.Activities.UI.Doctor.main.AdminMainActivity;
import com.example.heartdiseaseprediction.R;
import com.github.mikephil.charting.charts.BarChart;
import com.github.mikephil.charting.data.*;

import java.util.Calendar;
import java.util.List;
public class ChartActivity extends AppCompatActivity {

    TextView Date;
    ImageView Back_btn, imageButton;
    BarChart barChart;

    int year, month, day;
    String dateSelected;

    ChartViewModel viewModel;

    // ⚠️ lấy userId từ Intent (KHÔNG dùng session)
    String userId;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_chart);

        initView();

        viewModel = new ViewModelProvider(this).get(ChartViewModel.class);

        userId = getIntent().getStringExtra("USER_ID");

        setupDateDefault();
        observeData();
        loadData();
        setupClick();
    }

    private void initView() {
        Date = findViewById(R.id.date);
        Back_btn = findViewById(R.id.ButtonBack);
        imageButton = findViewById(R.id.dateicon);
        barChart = findViewById(R.id.bar_chart);
    }

    private void setupDateDefault() {
        Calendar calendar = Calendar.getInstance();

        year = calendar.get(Calendar.YEAR);
        month = calendar.get(Calendar.MONTH);
        day = calendar.get(Calendar.DAY_OF_MONTH);

        dateSelected = String.format("%04d%02d%02d", year, month + 1, day);
        Date.setText(day + "/" + (month + 1) + "/" + year);
    }

    private void loadData() {
        viewModel.loadChartData(userId, dateSelected);
    }

    private void observeData() {

        viewModel.getBarEntries().observe(this, list -> {
            if (list != null) {
                renderChart(list);
            }
        });

        viewModel.getError().observe(this, err -> {
            Toast.makeText(this, err, Toast.LENGTH_SHORT).show();
        });
    }

    private void renderChart(List<BarEntry> list) {

        BarDataSet dataSet = new BarDataSet(list, "Heart Beat");
        BarData data = new BarData(dataSet);

        barChart.setData(data);
        dataSet.setValueTextSize(14f);
        barChart.getDescription().setEnabled(false);
        barChart.invalidate();
    }

    private void setupClick() {

        Back_btn.setOnClickListener(v -> {
            startActivity(new Intent(this, AdminMainActivity.class));
        });

        imageButton.setOnClickListener(v -> {

            DatePickerDialog dialog = new DatePickerDialog(this,
                    (view, y, m, d) -> {

                        year = y;
                        month = m;
                        day = d;

                        dateSelected = String.format("%04d%02d%02d", year, month + 1, day);
                        Date.setText(day + "/" + (month + 1) + "/" + year);

                        loadData(); // reload chart
                    },
                    year, month, day);

            dialog.show();
        });
    }
}

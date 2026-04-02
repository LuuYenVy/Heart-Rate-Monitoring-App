package com.example.heartdiseaseprediction.Activities.UI.User.chart;

import androidx.fragment.app.Fragment;
import java.util.List;
import android.app.DatePickerDialog;
import android.graphics.Color;
import android.os.Bundle;
import androidx.lifecycle.ViewModelProvider;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;
import java.util.Calendar;
import com.example.heartdiseaseprediction.R;
import com.github.mikephil.charting.charts.BarChart;
import com.github.mikephil.charting.data.BarData;
import com.github.mikephil.charting.data.BarDataSet;
import com.github.mikephil.charting.data.BarEntry;
import com.google.firebase.auth.FirebaseAuth;
public class ChartFragment extends Fragment {

    private BarChart barChart;
    private TextView tvDate;
    private ImageView btnPickDate;

    private ChartViewModel viewModel;

    private int year, month, day;
    private String dateSelected;
    private String userId = FirebaseAuth.getInstance().getCurrentUser().getUid();

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container, Bundle savedInstanceState) {

        View view = inflater.inflate(R.layout.fragment_chart, container, false);

        barChart = view.findViewById(R.id.bar_chart);
        tvDate = view.findViewById(R.id.date);
        btnPickDate = view.findViewById(R.id.dateicon);

        viewModel = new ViewModelProvider(this).get(ChartViewModel.class);

        initDate();
        observeData();
        viewModel.loadChart(userId, dateSelected);

        btnPickDate.setOnClickListener(v -> showDatePicker());

        return view;
    }

    private void initDate() {
        Calendar cal = Calendar.getInstance();
        year = cal.get(Calendar.YEAR);
        month = cal.get(Calendar.MONTH);
        day = cal.get(Calendar.DAY_OF_MONTH);

        dateSelected = String.format("%04d%02d%02d", year, month + 1, day);
        tvDate.setText(day + "/" + (month + 1) + "/" + year);
    }

    private void observeData() {
        viewModel.getChartData().observe(getViewLifecycleOwner(), entries -> {
            if (entries != null) {
                renderChart(entries);
            }
        });

        viewModel.getError().observe(getViewLifecycleOwner(), err -> {
            Toast.makeText(getContext(), err, Toast.LENGTH_SHORT).show();
        });
    }

    private void renderChart(List<BarEntry> entries) {
        BarDataSet dataSet = new BarDataSet(entries, "Heart Beat");
        dataSet.setColor(Color.rgb(50,151,168));
        dataSet.setValueTextColor(Color.BLACK);
        dataSet.setValueTextSize(14f);

        BarData data = new BarData(dataSet);
        barChart.setData(data);
        barChart.getDescription().setEnabled(false);
        barChart.invalidate();
    }

    private void showDatePicker() {
        DatePickerDialog dialog = new DatePickerDialog(getContext(),
                (view, y, m, d) -> {
                    year = y;
                    month = m;
                    day = d;

                    dateSelected = String.format("%04d%02d%02d", year, month + 1, day);
                    tvDate.setText(day + "/" + (month + 1) + "/" + year);

                    viewModel.loadChart(userId, dateSelected);
                },
                year, month, day
        );

        dialog.show();
    }
}
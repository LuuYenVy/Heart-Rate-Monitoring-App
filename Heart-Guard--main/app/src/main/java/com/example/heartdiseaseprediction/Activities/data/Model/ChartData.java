package com.example.heartdiseaseprediction.Activities.data.Model;

import com.github.mikephil.charting.data.BarEntry;

import java.util.List;

public class ChartData {
    private List<BarEntry> entries;
    public ChartData(List<BarEntry> entries){
        this.entries=entries;
    }
    public List<BarEntry> getEntries(){
        return entries;
    }
}

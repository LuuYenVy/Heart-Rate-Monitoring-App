package com.example.heartdiseaseprediction.Activities.UI.User.chart;

import androidx.lifecycle.MutableLiveData;
import androidx.lifecycle.ViewModel;

import com.example.heartdiseaseprediction.Activities.data.Repository.ChartRepository;
import com.github.mikephil.charting.data.BarEntry;

import java.util.List;

public class ChartViewModel extends ViewModel {
    private ChartRepository repo = new ChartRepository();

    private MutableLiveData<List<BarEntry>> chartData= new MutableLiveData<>();

    private MutableLiveData<String> error= new MutableLiveData<>();
    public MutableLiveData<List<BarEntry>> getChartData() {
        return chartData;
    }

    public MutableLiveData<String> getError() {
        return error;
    }
    public void loadChart(String UserID, String date){
        repo.getChartData(UserID, date, new ChartRepository.Callback() {
            @Override
            public void onSuccess(List<BarEntry> entries) {
                chartData.postValue(entries);
            }

            @Override
            public void onError(String Error) {
                error.postValue(Error);
            }
        });
    }
}

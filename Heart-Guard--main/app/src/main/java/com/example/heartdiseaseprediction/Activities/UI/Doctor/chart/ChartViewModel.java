package com.example.heartdiseaseprediction.Activities.UI.Doctor.chart;

import androidx.lifecycle.MutableLiveData;
import androidx.lifecycle.ViewModel;

import com.github.mikephil.charting.data.BarEntry;
import com.google.firebase.database.*;

import java.util.ArrayList;
import java.util.List;

public class ChartViewModel extends ViewModel {

    private MutableLiveData<List<BarEntry>> barEntries = new MutableLiveData<>();
    private MutableLiveData<String> error = new MutableLiveData<>();

    public MutableLiveData<List<BarEntry>> getBarEntries() {
        return barEntries;
    }

    public MutableLiveData<String> getError() {
        return error;
    }

    public void loadChartData(String userId, String dateSelected) {

        DatabaseReference ref = FirebaseDatabase.getInstance()
                .getReference("users")
                .child(userId)
                .child("cardiacInfo");

        ref.addListenerForSingleValueEvent(new ValueEventListener() {
            @Override
            public void onDataChange(DataSnapshot snapshot) {

                List<BarEntry> list = new ArrayList<>();

                for (DataSnapshot child : snapshot.getChildren()) {

                    String key = child.getKey();

                    if (key != null && key.length() >= 10) {

                        String keyDate = key.substring(0, 8);

                        if (keyDate.equals(dateSelected)) {

                            String value = child.getValue(String.class);

                            if (value != null) {
                                float y = Float.parseFloat(value);
                                int heartBeat = Math.round(y);

                                int hour = Integer.parseInt(key.substring(8, 10));

                                list.add(new BarEntry(hour, heartBeat));
                            }
                        }
                    }
                }

                barEntries.setValue(list);
            }

            @Override
            public void onCancelled(DatabaseError errorDb) {
                error.setValue(errorDb.getMessage());
            }
        });
    }
}
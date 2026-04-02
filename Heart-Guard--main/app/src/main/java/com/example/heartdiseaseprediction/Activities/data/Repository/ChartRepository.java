package com.example.heartdiseaseprediction.Activities.data.Repository;

import androidx.annotation.NonNull;

import com.github.mikephil.charting.data.BarEntry;
import com.google.firebase.database.DataSnapshot;
import com.google.firebase.database.DatabaseError;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;
import com.google.firebase.database.ValueEventListener;

import java.util.ArrayList;
import java.util.List;

public class ChartRepository {
    public interface Callback{
        void onSuccess(List<BarEntry> entries);
        void onError(String Error);
    }
    public void getChartData(String userID, String dataSelected,Callback callback){
        DatabaseReference ref= FirebaseDatabase.getInstance()
                .getReference("users")
                .child(userID)
                .child("cardiacInfo");
        ref.addValueEventListener(new ValueEventListener() {
            @Override
            public void onDataChange(@NonNull DataSnapshot snapshot) {
                List<BarEntry> list = new ArrayList<>();
                for(DataSnapshot child: snapshot.getChildren()){
                    String key= child.getKey();
                    if(key!=null && key.length() >=10){
                        String keyDate=key.substring(0,8);
                        if(keyDate.equals(dataSelected)){
                            String value= child.getValue(String.class);
                            float y =Float.parseFloat(value);
                            int hour = Integer.parseInt(key.substring(8,10));
                            list.add(new BarEntry(hour,y));
                        }
                    }
                }
                callback.onSuccess(list);
            }

            @Override
            public void onCancelled(@NonNull DatabaseError error) {
                callback.onError(error.getMessage());
            }
        });
    }
}

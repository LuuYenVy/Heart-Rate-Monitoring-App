package com.example.heartdiseaseprediction.Activities.data.Repository;

import androidx.annotation.NonNull;

import com.example.heartdiseaseprediction.Activities.data.Model.HeartResult;
import com.google.firebase.auth.FirebaseAuth;
import com.google.firebase.database.DataSnapshot;
import com.google.firebase.database.DatabaseError;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;
import com.google.firebase.database.ValueEventListener;

import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;
import java.util.Random;

public class HeartRepository {
    private DatabaseReference databaseReference;
    public HeartRepository(){
        String userId = FirebaseAuth.getInstance().getCurrentUser().getUid();
        databaseReference = FirebaseDatabase.getInstance()
                .getReference("users")
                .child(userId)
                .child("heartrate")
                .child("latest");
    }
    public void listenHeartRate(HeartCallback callback) {
        databaseReference.addValueEventListener(new ValueEventListener() {
            @Override
            public void onDataChange(@NonNull DataSnapshot snapshot) {
                try {
                    String beatValue = snapshot.child("beat").getValue(String.class);

                    if (beatValue == null) {
                        callback.onError("No data");
                        return;
                    }

                    String time = new SimpleDateFormat("HH:mm:ss",
                            Locale.getDefault()).format(new Date());

                    boolean isDanger = predict(Integer.parseInt(beatValue));

                    callback.onSuccess(new HeartResult(beatValue, time, isDanger));

                } catch (Exception e) {
                    callback.onError(e.getMessage());
                }
            }

            @Override
            public void onCancelled(@NonNull DatabaseError error) {
                callback.onError(error.getMessage());
            }
        });
    }
    private boolean predict( int heartRate){
        return heartRate >120;
    }
    public  interface HeartCallback{
        void onSuccess(HeartResult result);
        void onError(String error);
    }
    //Demo dâta
    public void pushFakeData() {
        String userId = FirebaseAuth.getInstance().getCurrentUser().getUid();

        DatabaseReference rootRef = FirebaseDatabase.getInstance()
                .getReference("users")
                .child(userId);

        String time = new SimpleDateFormat("HH:mm:ss",
                Locale.getDefault()).format(new Date());

        int beat = 60 + new Random().nextInt(40); // random 60-100

        // 🔹 1. Lưu latest
        rootRef.child("heartrate").child("latest")
                .setValue(new HeartResult(String.valueOf(beat), time, beat > 120));

        // 🔹 2. Lưu history cho chart
        String key = new SimpleDateFormat("yyyyMMddHHmmss",
                Locale.getDefault()).format(new Date());

        rootRef.child("cardiacInfo")
                .child(key)
                .setValue(String.valueOf(beat));
    }
}

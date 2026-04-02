package com.example.heartdiseaseprediction.Activities.data.Repository;

import android.util.Log;

import com.example.heartdiseaseprediction.Activities.data.Model.Medicine;
import com.google.firebase.database.DataSnapshot;
import com.google.firebase.database.DatabaseError;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;
import com.google.firebase.database.ValueEventListener;
import com.google.firebase.firestore.FirebaseFirestore;
import com.google.firebase.firestore.QueryDocumentSnapshot;

import java.util.ArrayList;
import java.util.List;

public class PrescriptionRepository {

    private final DatabaseReference database = FirebaseDatabase.getInstance().getReference();
    public void getMedicines(OnMedicinesLoaded callback) {

        FirebaseFirestore db = FirebaseFirestore.getInstance();

        db.collection("medicines")
                .get()
                .addOnSuccessListener(queryDocumentSnapshots -> {

                    List<String> list = new ArrayList<>();

                    for (QueryDocumentSnapshot doc : queryDocumentSnapshots) {
                        String name = doc.getString("name");
                        if (name != null) {
                            list.add(name);
                        }
                    }

                    callback.onSuccess(list);
                })
                .addOnFailureListener(e -> {
                    e.printStackTrace();
                });
    }

    public void saveMedicines(String appointmentId, List<Medicine> medicines) {

        if (appointmentId == null || appointmentId.isEmpty()) {
            Log.e("ERROR", "appointmentId is null");
            return;
        }

        DatabaseReference ref = FirebaseDatabase.getInstance()
                .getReference("appointments")
                .child(appointmentId)
                .child("medicines");

        for (Medicine m : medicines) {

            String key = ref.push().getKey();

            if (key == null) continue;

            ref.child(key).setValue(m)
                    .addOnSuccessListener(aVoid ->
                            Log.d("Firebase", "Saved medicine"))
                    .addOnFailureListener(e ->
                            Log.e("Firebase", "Error: " + e.getMessage()));
        }
    }

    public interface OnMedicinesLoaded {
        void onSuccess(List<String> list);
    }
}

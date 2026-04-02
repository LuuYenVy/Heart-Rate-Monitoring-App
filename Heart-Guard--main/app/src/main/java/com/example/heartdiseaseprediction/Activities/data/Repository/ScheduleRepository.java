package com.example.heartdiseaseprediction.Activities.data.Repository;

import com.example.heartdiseaseprediction.Activities.data.Model.Appointment;

import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;
import com.google.firebase.firestore.FirebaseFirestore;
import com.google.firebase.firestore.DocumentSnapshot;
import java.util.ArrayList;
import java.util.List;

public class ScheduleRepository {

    private DatabaseReference serviceRef = FirebaseDatabase.getInstance().getReference("service");
    private DatabaseReference appointmentRef = FirebaseDatabase.getInstance().getReference("appointments");

    // callback doctor
    public interface DoctorCallback {
        void onSuccess(List<String> list);
        void onError(String error);
    }

    // callback create appointment
    public interface CreateCallback {
        void onSuccess();
        void onError(String error);
    }

    public void getDoctors(String selectedService, DoctorCallback callback) {

        FirebaseFirestore db = FirebaseFirestore.getInstance();

        db.collection("DoctorCollection")
                .whereEqualTo("service", selectedService)
                .get()
                .addOnSuccessListener(snapshot -> {

                    List<String> doctors = new ArrayList<>();

                    for (DocumentSnapshot doc : snapshot.getDocuments()) {
                        String name = doc.getString("name");
                        if (name != null) {
                            doctors.add(name);
                        }
                    }

                    callback.onSuccess(doctors);
                })
                .addOnFailureListener(e -> {
                    callback.onError(e.getMessage());
                });
    }
    public void createAppointment(String id, Appointment appointment, CreateCallback callback) {
        appointmentRef.child(id).setValue(appointment)
                .addOnSuccessListener(unused -> callback.onSuccess())
                .addOnFailureListener(e -> callback.onError(e.getMessage()));
    }
}
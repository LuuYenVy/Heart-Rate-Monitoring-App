package com.example.heartdiseaseprediction.Activities.data.Repository;


import com.example.heartdiseaseprediction.Activities.data.Model.Appointment;
import com.google.firebase.database.*;
import com.google.firebase.firestore.FirebaseFirestore;

import java.util.ArrayList;
import java.util.List;
public class AdminAppointmentRepository {

    public interface Callback {
        void onSuccess(List<Appointment> list);
        void onError(String error);
    }

    public void getAppointments(String doctorName, String status, Callback callback) {

        DatabaseReference ref = FirebaseDatabase.getInstance().getReference("appointments");

        ref.orderByChild("status").equalTo(status)
                .addListenerForSingleValueEvent(new ValueEventListener() {
                    @Override
                    public void onDataChange(DataSnapshot snapshot) {

                        List<Appointment> list = new ArrayList<>();

                        for (DataSnapshot item : snapshot.getChildren()) {

                            String doctor = item.child("doctor").getValue(String.class);

                            if (doctorName != null && doctorName.equals(doctor)) {
                                Appointment ap = item.getValue(Appointment.class);

                                if (ap != null) {
                                    ap.setId(item.getKey());
                                    list.add(ap);
                                }
                            }
                        }

                        callback.onSuccess(list);
                    }

                    @Override
                    public void onCancelled(DatabaseError error) {
                        callback.onError(error.getMessage());
                    }
                });
    }
    public void getAppointmentById(String appointmentId, SingleAppointmentCallback callback) {

        DatabaseReference ref = FirebaseDatabase.getInstance()
                .getReference("appointments")
                .child(appointmentId);

        ref.addListenerForSingleValueEvent(new ValueEventListener() {
            @Override
            public void onDataChange(DataSnapshot snapshot) {

                if (snapshot.exists()) {

                    Appointment ap = snapshot.getValue(Appointment.class);

                    if (ap != null) {
                        ap.setId(snapshot.getKey());
                        callback.onSuccess(ap);
                    } else {
                        callback.onError("Appointment null");
                    }

                } else {
                    callback.onError("Appointment not found");
                }
            }

            @Override
            public void onCancelled(DatabaseError error) {
                callback.onError(error.getMessage());
            }
        });
    }
    public interface SingleAppointmentCallback {
        void onSuccess(Appointment appointment);
        void onError(String error);
    }
}
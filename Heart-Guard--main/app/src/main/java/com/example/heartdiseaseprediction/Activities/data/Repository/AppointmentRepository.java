package com.example.heartdiseaseprediction.Activities.data.Repository;

import androidx.annotation.NonNull;

import com.example.heartdiseaseprediction.Activities.data.Model.Appointment;
import com.google.firebase.database.DataSnapshot;
import com.google.firebase.database.DatabaseError;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;
import com.google.firebase.database.ValueEventListener;
import com.google.firebase.firestore.FirebaseFirestore;
import com.google.firebase.firestore.DocumentSnapshot;

import java.util.ArrayList;
import java.util.List;

public class AppointmentRepository {

    private DatabaseReference ref = FirebaseDatabase.getInstance().getReference("appointments");
    private FirebaseFirestore db = FirebaseFirestore.getInstance();

    public interface ServiceCallback {
        void onSuccess(List<String> services);
        void onError(String error);
    }

    public void getAppointments(String userId, String status, Callback callback) {

        ref.orderByChild("status").equalTo(status)
                .addListenerForSingleValueEvent(new ValueEventListener() {
                    @Override
                    public void onDataChange(@NonNull DataSnapshot snapshot) {

                        List<Appointment> list = new ArrayList<>();

                        for (DataSnapshot item : snapshot.getChildren()) {

                            String uid = item.child("user").child("userID").getValue(String.class);

                            if (uid != null && uid.equals(userId)) {

                                list.add(new Appointment(
                                        item.getKey(),
                                        item.child("service").getValue(String.class),
                                        item.child("date").getValue(String.class),
                                        status
                                ));
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

    public void getAppointmentById(String id, DetailAppointmentCallback callback) {
        ref.child(id).addListenerForSingleValueEvent(new ValueEventListener() {
            @Override
            public void onDataChange(DataSnapshot snapshot) {
                if(snapshot.exists()){
                    Appointment ap = snapshot.getValue(Appointment.class);
                    callback.onSuccess(ap);
                }else{
                    callback.onError("Not found");
                }
            }

            @Override
            public void onCancelled(DatabaseError error) {
                callback.onError(error.getMessage());
            }
        });
    }
    public void getServices(ServiceCallback callback) {
        db.collection("Service")
                .get()
                .addOnSuccessListener(snapshot -> {

                    List<String> list = new ArrayList<>();

                    for (DocumentSnapshot doc : snapshot.getDocuments()) {
                        String name = doc.getString("name");
                        if (name != null) {
                            list.add(name);
                        }
                    }

                    callback.onSuccess(list);
                })
                .addOnFailureListener(e -> {
                    callback.onError(e.getMessage());
                });
    }

    public interface Callback {
        void onSuccess(List<Appointment> list);
        void onError(String error);
    }
    public interface DetailAppointmentCallback{
        void onSuccess(Appointment appointment);
        void onError(String error);
    }
}


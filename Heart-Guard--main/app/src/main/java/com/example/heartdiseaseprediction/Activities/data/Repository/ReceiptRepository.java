package com.example.heartdiseaseprediction.Activities.data.Repository;

import com.example.heartdiseaseprediction.Activities.data.Model.Medicine;
import com.google.firebase.database.*;

import java.util.ArrayList;
import java.util.List;

public class ReceiptRepository {

    public interface Callback {
        void onSuccess(List<Medicine> list);
        void onError(String error);
    }

    public void getMedicines(String appointmentId, Callback callback) {
        DatabaseReference ref = FirebaseDatabase.getInstance()
                .getReference("appointments")
                .child(appointmentId)
                .child("medicines");

        ref.addListenerForSingleValueEvent(new ValueEventListener() {
            @Override
            public void onDataChange(DataSnapshot snapshot) {
                List<Medicine> list = new ArrayList<>();

                for (DataSnapshot item : snapshot.getChildren()) {
                    String name = item.child("name").getValue(String.class);
                    String routine = item.child("routine").getValue(String.class);
                    String amount = item.child("amount").getValue(String.class);

                    list.add(new Medicine(name, routine, amount));
                }

                callback.onSuccess(list);
            }

            @Override
            public void onCancelled(DatabaseError error) {
                callback.onError(error.getMessage());
            }
        });
    }
}

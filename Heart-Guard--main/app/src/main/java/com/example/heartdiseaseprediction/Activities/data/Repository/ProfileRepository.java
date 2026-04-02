package com.example.heartdiseaseprediction.Activities.data.Repository;
import com.example.heartdiseaseprediction.Activities.data.Model.User;
import com.google.firebase.database.DataSnapshot;
import com.google.firebase.database.DatabaseError;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;
import com.google.firebase.database.ValueEventListener;

public class ProfileRepository {

    public interface Callback {
        void onSuccess(User user);
        void onError(String error);
    }

    public void getUser(String userId, Callback callback) {
        DatabaseReference ref = FirebaseDatabase.getInstance()
                .getReference("users")
                .child(userId);

        ref.addListenerForSingleValueEvent(new ValueEventListener() {
            @Override
            public void onDataChange(DataSnapshot snapshot) {
                if (snapshot.exists()) {
                    User user = snapshot.getValue(User.class);
                    callback.onSuccess(user);
                } else {
                    callback.onError("User not found");
                }
            }

            @Override
            public void onCancelled(DatabaseError error) {
                callback.onError(error.getMessage());
            }
        });
    }

    public void updateUser(String userId, String age, String height, String weight, Callback callback) {
        DatabaseReference ref = FirebaseDatabase.getInstance()
                .getReference("users")
                .child(userId);

        ref.child("age").setValue(age);
        ref.child("height").setValue(height);
        ref.child("weight").setValue(weight);

        // gọi lại để lấy user mới
        getUser(userId, callback);
    }
}
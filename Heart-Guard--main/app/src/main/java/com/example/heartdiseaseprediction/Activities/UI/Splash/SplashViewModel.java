package com.example.heartdiseaseprediction.Activities.UI.Splash;

import android.content.Context;
import android.content.SharedPreferences;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;
import androidx.lifecycle.ViewModel;

import com.example.heartdiseaseprediction.Activities.UI.Auth.LoginActivity;
import com.example.heartdiseaseprediction.Activities.UI.Doctor.main.AdminMainActivity;
import com.example.heartdiseaseprediction.Activities.UI.User.MainActivity;
import com.google.firebase.auth.FirebaseAuth;
import com.google.firebase.auth.FirebaseUser;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;

public class SplashViewModel extends ViewModel {

    private final MutableLiveData<Class<?>> navigateTo = new MutableLiveData<>();

    public LiveData<Class<?>> getNavigateTo() {
        return navigateTo;
    }

    public void checkUser(Context context) {

        FirebaseUser currentUser = FirebaseAuth.getInstance().getCurrentUser();
        DatabaseReference ref = FirebaseDatabase.getInstance()
                .getReference("users")
                .child(currentUser.getUid());

        ref.child("role").get().addOnSuccessListener(snapshot -> {
            String role = snapshot.getValue(String.class);

            if ("doctor".equals(role)) {
                navigateTo.setValue(AdminMainActivity.class);
            } else {
                navigateTo.setValue(MainActivity.class);
            }
        });

    }
}

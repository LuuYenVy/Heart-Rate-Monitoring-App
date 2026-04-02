package com.example.heartdiseaseprediction.Activities.data.Repository;
import android.util.Log;

import androidx.annotation.NonNull;
import androidx.lifecycle.MutableLiveData;
import com.example.heartdiseaseprediction.Activities.data.Model.LoginResult;
import com.example.heartdiseaseprediction.Activities.data.Model.RegisterResult;
import com.example.heartdiseaseprediction.Activities.data.Model.User;
import com.google.firebase.auth.AuthResult;
import com.google.firebase.auth.FirebaseAuth;
import com.google.firebase.auth.FirebaseUser;
import com.google.firebase.database.DatabaseError;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;
import com.google.firebase.database.ValueEventListener;
import com.google.firebase.database.DataSnapshot;
import com.google.firebase.firestore.FirebaseFirestore;

import java.util.HashMap;
import java.util.Map;

public class AuthRepository {
    private FirebaseAuth mAuth = FirebaseAuth.getInstance();
    private  DatabaseReference userRef= FirebaseDatabase.getInstance().getReference("users");

//    public void login(String username, String password, MutableLiveData<LoginResult> resultLiveData){
//        userRef.orderByChild("username")
//                .equalTo(username)
//                .addListenerForSingleValueEvent(new ValueEventListener() {
//                    @Override
//                    public void onDataChange(@NonNull DataSnapshot snapshot) {
//                        if(!snapshot.exists()){
//                            resultLiveData.setValue(new LoginResult(false,false,"User not found"));
//                            return;
//                        }
//                        for( DataSnapshot usersnap : snapshot.getChildren()){
//                            User user = usersnap.getValue(User.class);
//                            if(user == null){
//                                resultLiveData.setValue(
//                                        new LoginResult(false,false,"User data error")
//                                );
//                                return;
//                            }
//                            break;
//                        }
//                    }
//
//                    @Override
//                    public void onCancelled(@NonNull DatabaseError error) {
//                        resultLiveData.setValue(new LoginResult(false,false,error.getMessage()));
//                    }
//                });
//
//
//    }
    public void saveUserInfo(String gender, String weight, String height, String age,
                             MutableLiveData<LoginResult> resultLiveData){

        FirebaseUser currentUser = FirebaseAuth.getInstance().getCurrentUser();

        if(currentUser == null){
            resultLiveData.setValue(new LoginResult(false, null, "User not logged in"));
            return;
        }

        String userID = currentUser.getUid();

        DatabaseReference userRef = FirebaseDatabase.getInstance()
                .getReference("users")
                .child(userID);

        Map<String, Object> updates = new HashMap<>();
        updates.put("gender", gender);
        updates.put("weight", weight);
        updates.put("height", height);
        updates.put("age", age);

        userRef.updateChildren(updates).addOnCompleteListener(task -> {
            if(task.isSuccessful()){
                resultLiveData.setValue(new LoginResult(true, "user", "Success"));
            } else {
                resultLiveData.setValue(new LoginResult(false, null, task.getException().getMessage()));
            }
        });
    }
    public void loginWithEmail(String email, String password, MutableLiveData<LoginResult> resultLiveData){

        mAuth.signInWithEmailAndPassword(email, password)
                .addOnCompleteListener(task -> {

                    if(!task.isSuccessful()){
                        resultLiveData.setValue(new LoginResult(false,null,task.getException().getMessage()));
                        return;
                    }

                    FirebaseUser firebaseUser = mAuth.getCurrentUser();

                    if(firebaseUser == null){
                        resultLiveData.setValue(new LoginResult(false,null,"User null"));
                        return;
                    }

                    String userId = firebaseUser.getUid();

                    DatabaseReference userRef = FirebaseDatabase.getInstance()
                            .getReference("users")
                            .child(userId);

                    userRef.addListenerForSingleValueEvent(new ValueEventListener() {
                        @Override
                        public void onDataChange(DataSnapshot snapshot) {

                            if (!snapshot.exists()) {
                                resultLiveData.setValue(new LoginResult(false,null,"User profile not found"));
                                return;
                            }

                            User user = snapshot.getValue(User.class);

                            if (user == null) {
                                resultLiveData.setValue(new LoginResult(false,null,"User data null"));
                                return;
                            }

                            user.setUserID(userId);

                            String role = user.getRole();

                            resultLiveData.setValue(
                                    new LoginResult(true, role, "Success", user)
                            );
                        }

                        @Override
                        public void onCancelled(DatabaseError error) {
                            resultLiveData.setValue(new LoginResult(false,null,error.getMessage()));
                        }
                    });
                });
    }
    public void register(String email, String password, String username, RegisterCallback callback){
        mAuth.createUserWithEmailAndPassword(email,password)
                .addOnCompleteListener(task -> {
                    if(!task.isSuccessful()){
                        callback.onResult(new RegisterResult(false, task.getException().getMessage()));
                        return;
                    }

                    FirebaseUser user = mAuth.getCurrentUser();

                    if(user == null){
                        callback.onResult(new RegisterResult(false, "User null"));
                        return;
                    }
                    User userobj = new User(email, username);
                    userobj.setRole("user");

                    userRef.child(user.getUid()).setValue(userobj)
                            .addOnCompleteListener(dbtask -> {
                                if(dbtask.isSuccessful()){
                                    callback.onResult(new RegisterResult(true,"Success"));
                                } else {
                                    callback.onResult(new RegisterResult(false, dbtask.getException().getMessage()));
                                }
                            });
                });
    }
    public interface RegisterCallback{
        void onResult(RegisterResult result);
    }
}

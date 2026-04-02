package com.example.heartdiseaseprediction.Activities.UI.Auth;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;
import androidx.lifecycle.ViewModel;
import com.example.heartdiseaseprediction.Activities.data.Model.LoginResult;
import com.example.heartdiseaseprediction.Activities.data.Model.RegisterResult;
import com.example.heartdiseaseprediction.Activities.data.Repository.AuthRepository;

public class AuthViewModel extends ViewModel {
    private AuthRepository repository;

    public AuthRepository getRepository() {
        return repository;
    }

    private MutableLiveData<LoginResult> loginResult = new MutableLiveData<>();

    private MutableLiveData<RegisterResult> registerResult = new MutableLiveData<>();
    public MutableLiveData<LoginResult> getSaveInfoResult() {
        return saveInfoResult;
    }

    private MutableLiveData<LoginResult> saveInfoResult = new MutableLiveData<>();
    public AuthViewModel(){
        repository= new AuthRepository();
    }
    public LiveData<LoginResult> getLoginResult(){
        return loginResult;
    }
    public LiveData<LoginResult> saveUserInfoResult(){
        return  saveInfoResult;
    }

    public LiveData<RegisterResult> getregisterResult(){
        return  registerResult;
    }
    public void loginWithEmail(String email, String password){
        repository.loginWithEmail(email,password,loginResult);
    }
    public void saveUserInfo(String gender, String weight, String height, String age){
        repository.saveUserInfo(gender,weight,height, age,saveInfoResult);
    }
    public void register(String email, String password, String confirm, String username){
        if(email.isEmpty()||password.isEmpty()||username.isEmpty()){
            registerResult.setValue(new RegisterResult(false,"There are empty fields."));
            return;
        }
        repository.register(email,password,username,result -> {
            registerResult.postValue(result);
        });
    }
}

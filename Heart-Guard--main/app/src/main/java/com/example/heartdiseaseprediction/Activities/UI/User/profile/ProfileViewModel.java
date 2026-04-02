package com.example.heartdiseaseprediction.Activities.UI.User.profile;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;
import androidx.lifecycle.ViewModel;
import com.example.heartdiseaseprediction.Activities.data.Model.User;
import com.example.heartdiseaseprediction.Activities.data.Repository.ProfileRepository;

public class ProfileViewModel extends ViewModel {

    private MutableLiveData<User> userLiveData = new MutableLiveData<>();
    private MutableLiveData<String> message = new MutableLiveData<>();

    private ProfileRepository repo = new ProfileRepository();

    public LiveData<User> getUser() {
        return userLiveData;
    }

    public LiveData<String> getMessage() {
        return message;
    }

    public void loadUser(String userId) {
        repo.getUser(userId, new ProfileRepository.Callback() {
            @Override
            public void onSuccess(User user) {
                userLiveData.postValue(user);
            }

            @Override
            public void onError(String error) {
                message.postValue(error);
            }
        });
    }

    public void updateUser(String userId, String age, String height, String weight) {
        repo.updateUser(userId, age, height, weight, new ProfileRepository.Callback() {
            @Override
            public void onSuccess(User user) {
                userLiveData.postValue(user);
                message.postValue("Update success");
            }

            @Override
            public void onError(String error) {
                message.postValue(error);
            }
        });
    }
}
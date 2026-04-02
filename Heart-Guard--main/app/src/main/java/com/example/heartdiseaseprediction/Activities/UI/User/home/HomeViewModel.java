package com.example.heartdiseaseprediction.Activities.UI.User.home;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;
import androidx.lifecycle.ViewModel;

import com.example.heartdiseaseprediction.Activities.data.Model.HeartResult;
import com.example.heartdiseaseprediction.Activities.data.Repository.HeartRepository;

public class HomeViewModel extends ViewModel {

    private MutableLiveData<HeartResult> heartData = new MutableLiveData<>();
    private HeartRepository repository = new HeartRepository();

    public LiveData<HeartResult> getHeartData() {
        return heartData;
    }

    public void startListening() {
        repository.listenHeartRate(new HeartRepository.HeartCallback() {
            @Override
            public void onSuccess(HeartResult result) {
                heartData.postValue(result);
            }

            @Override
            public void onError(String error) {
            }
        });
    }
    public void pushFakeData(){
        repository.pushFakeData();
    }
}

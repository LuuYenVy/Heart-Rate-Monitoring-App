package com.example.heartdiseaseprediction.Activities.UI.User.chatbot;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;
import androidx.lifecycle.ViewModel;

import com.example.heartdiseaseprediction.Activities.data.Model.AIResponse;
import com.example.heartdiseaseprediction.Activities.data.Repository.ChatRepository;

import java.util.List;

import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

public class ChatViewModel extends ViewModel {

    private ChatRepository repository;

    private MutableLiveData<String> responseLiveData = new MutableLiveData<>();

    public ChatViewModel() {
        repository = new ChatRepository();
    }

    public LiveData<String> getResponse() {
        return responseLiveData;
    }

    public void sendToAI(List<Integer> heartRate, String question) {

        repository.analyze(heartRate, question, new Callback<AIResponse>() {
            @Override
            public void onResponse(Call<AIResponse> call, Response<AIResponse> response) {
                if (response.isSuccessful() && response.body() != null) {
                    responseLiveData.postValue(response.body().getResult());
                } else {
                    responseLiveData.postValue("Error response");
                }
            }

            @Override
            public void onFailure(Call<AIResponse> call, Throwable t) {
                responseLiveData.postValue("Failure: " + t.getMessage());
            }
        });
    }
}
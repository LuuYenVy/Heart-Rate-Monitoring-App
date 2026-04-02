package com.example.heartdiseaseprediction.Activities.data.Repository;

import com.example.heartdiseaseprediction.Activities.data.Model.AIRequest;
import com.example.heartdiseaseprediction.Activities.data.Model.AIResponse;
import com.example.heartdiseaseprediction.Activities.data.Remote.ApiService;
import com.example.heartdiseaseprediction.Activities.data.Remote.RetrofitClient;

import java.util.List;

import retrofit2.Callback;

public class ChatRepository {

    private ApiService apiService;

    public ChatRepository() {
        apiService = RetrofitClient.getInstance().create(ApiService.class);
    }

    public void analyze(List<Integer> heartRate, String question, Callback<AIResponse> callback) {

        AIRequest request = new AIRequest(heartRate, question);

        apiService.analyze(request).enqueue(callback);
    }
}
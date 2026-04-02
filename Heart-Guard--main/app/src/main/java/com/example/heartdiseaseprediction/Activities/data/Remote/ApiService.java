package com.example.heartdiseaseprediction.Activities.data.Remote;

import com.example.heartdiseaseprediction.Activities.data.Model.AIRequest;
import com.example.heartdiseaseprediction.Activities.data.Model.AIResponse;

import retrofit2.Call;
import retrofit2.http.Body;
import retrofit2.http.POST;

public interface ApiService {

    @POST("/analyze")
    Call<AIResponse> analyze(@Body AIRequest request);
}

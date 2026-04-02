package com.example.heartdiseaseprediction.Activities.data.Model;

import java.util.List;

public class AIRequest {
    private List<Integer> heartRate;
    private String question;

    public AIRequest(List<Integer> heartRate, String question) {
        this.heartRate = heartRate;
        this.question = question;
    }
}

package com.example.heartdiseaseprediction.Activities.data.Model;

public class RegisterResult {
    private boolean success;
    private String message;

    public RegisterResult(boolean success, String message) {
        this.success = success;
        this.message = message;
    }

    public boolean isSuccess() { return success; }
    public String getMessage() { return message; }
}
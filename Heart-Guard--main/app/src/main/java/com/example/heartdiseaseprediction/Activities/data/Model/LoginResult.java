package com.example.heartdiseaseprediction.Activities.data.Model;

public class LoginResult {
    private boolean success;


    private String role;
    private User user;

    private String message;

    public LoginResult(boolean success, String role, String message, User user) {
        this.success = success;
        this.role = role;
        this.message = message;
        this.user=user;
    }

    public LoginResult(boolean success, String role, String message) {
        this.success = success;
        this.role=role;
        this.message = message;
    }

    public boolean isSuccess() { return success; }
    public String getRole() {
        return role;
    }

    public void setRole(String role) {
        this.role = role;
    }
    public String getMessage() { return message; }
    public User getUser() {
        return user;
    }
}
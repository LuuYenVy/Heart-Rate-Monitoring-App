package com.example.heartdiseaseprediction.Activities.data.Model;

public class User {
    private String userID;
    private String email;
    private String username;
    private String age;
    private String height;
    private  String weight;
    private String gender;
    private String role;

    public User() {
    }

    public User(String userID,String email, String username, String age, String height, String weight,String gender) {
        this.email = email;
        this.username = username;
        this.age=age;
        this.height=height;
        this.weight=weight;
        this.gender=gender;
        this.userID=userID;
    }
    public User(String email,  String username) {
        this.email=email;
        this.username = username;
    }
    public User(String age, String height, String weight,String gender) {
        this.age=age;
        this.height=height;
        this.weight = weight;
        this.gender=gender;
    }


    public String getEmail() {
        return email;
    }

    public void setEmail(String email) {
        this.email = email;
    }


    public String getUsername() {
        return username;
    }

    public void setUsername(String username) {
        this.username = username;
    }
    public String getAge() {
        return age;
    }

    public void setAge(String age) {
        this.age = age;
    }

    public String getHeight() {
        return height;
    }

    public void setHeight(String height) {
        this.height = height;
    }

    public String getWeight() {
        return weight;
    }

    public void setWeight(String weight) {
        this.weight = weight;
    }
    public String getGender() {
        return gender;
    }

    public void setGender(String gender) {
        this.gender = gender;
    }
    public String getUserID() {
        return userID;
    }

    public void setUserID(String userID) {
        this.userID = userID;
    }
    public String getRole() {
        return this.role;
    }
    public void setRole(String role) {
        this.role = role;
    }
}


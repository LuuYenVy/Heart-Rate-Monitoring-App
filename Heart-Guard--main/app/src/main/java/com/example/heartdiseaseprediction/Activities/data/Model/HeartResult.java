package com.example.heartdiseaseprediction.Activities.data.Model;

public class HeartResult {
    private String beat;
    private String time;
    private boolean isDanger;

    public HeartResult(String beat, String time, boolean isDanger) {
        this.beat = beat;
        this.time = time;
        this.isDanger = isDanger;
    }

    public String getBeat() { return beat; }
    public String getTime() { return time; }
    public boolean isDanger() { return isDanger; }
}

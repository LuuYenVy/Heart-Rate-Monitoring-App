package com.example.heartdiseaseprediction.Activities.UI.Doctor.appointment;

import androidx.lifecycle.MutableLiveData;
import androidx.lifecycle.ViewModel;

import com.example.heartdiseaseprediction.Activities.data.Model.Appointment;
import com.example.heartdiseaseprediction.Activities.data.Model.User;
import com.google.firebase.database.*;

public class DoctorAppointmentViewModel extends ViewModel {

    private MutableLiveData<Appointment> appointmentLiveData = new MutableLiveData<>();
    private MutableLiveData<User> userLiveData = new MutableLiveData<>();
    private MutableLiveData<String> symptom = new MutableLiveData<>();
    private MutableLiveData<String> medicalHistory = new MutableLiveData<>();
    private MutableLiveData<String> diagnostic = new MutableLiveData<>();
    private MutableLiveData<String> doctorName = new MutableLiveData<>();
    private MutableLiveData<String> error = new MutableLiveData<>();

    public MutableLiveData<Appointment> getAppointment() {
        return appointmentLiveData;
    }

    public MutableLiveData<User> getUser() {
        return userLiveData;
    }

    public MutableLiveData<String> getSymptom() {
        return symptom;
    }

    public MutableLiveData<String> getMedicalHistory() {
        return medicalHistory;
    }

    public MutableLiveData<String> getDiagnostic() {
        return diagnostic;
    }

    public MutableLiveData<String> getDoctorName() {
        return doctorName;
    }

    public MutableLiveData<String> getError() {
        return error;
    }

    // load data từ Firebase
    public void loadAppointmentDetail(String appointmentId) {
        DatabaseReference ref = FirebaseDatabase.getInstance()
                .getReference("appointments")
                .child(appointmentId);

        ref.addListenerForSingleValueEvent(new ValueEventListener() {
            @Override
            public void onDataChange(DataSnapshot snapshot) {

                if (!snapshot.exists()) {
                    error.setValue("Appointment not found");
                    return;
                }

                Appointment appointment = snapshot.getValue(Appointment.class);
                appointmentLiveData.setValue(appointment);

                // lấy user
                User user = snapshot.child("user").getValue(User.class);
                userLiveData.setValue(user);

                symptom.setValue(snapshot.child("Sypptom").getValue(String.class));
                medicalHistory.setValue(snapshot.child("MedicalHistory").getValue(String.class));
                diagnostic.setValue(snapshot.child("Diagnostic").getValue(String.class));
                doctorName.setValue(snapshot.child("doctor").getValue(String.class));
            }

            @Override
            public void onCancelled(DatabaseError errorDb) {
                error.setValue(errorDb.getMessage());
            }
        });
    }
}
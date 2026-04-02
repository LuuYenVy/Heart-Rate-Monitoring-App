package com.example.heartdiseaseprediction.Activities.UI.Doctor.main;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;
import androidx.lifecycle.ViewModel;

import com.example.heartdiseaseprediction.Activities.data.Model.Appointment;
import com.example.heartdiseaseprediction.Activities.data.Repository.AdminAppointmentRepository;

import java.util.List;

public class DoctorViewModel extends ViewModel {

    private MutableLiveData<List<Appointment>> incoming = new MutableLiveData<>();
    private MutableLiveData<List<Appointment>> history = new MutableLiveData<>();
    private MutableLiveData<String> error = new MutableLiveData<>();

    private AdminAppointmentRepository repo = new AdminAppointmentRepository();

    public LiveData<List<Appointment>> getIncoming() {
        return incoming;
    }

    public LiveData<List<Appointment>> getHistory() {
        return history;
    }

    public LiveData<String> getError() {
        return error;
    }

    public void loadData(String doctorName) {

        repo.getAppointments(doctorName, "incoming", new AdminAppointmentRepository.Callback() {
            @Override
            public void onSuccess(List<Appointment> list) {
                incoming.setValue(list);
            }

            @Override
            public void onError(String err) {
                error.setValue(err);
            }
        });

        repo.getAppointments(doctorName, "finish", new AdminAppointmentRepository.Callback() {
            @Override
            public void onSuccess(List<Appointment> list) {
                history.setValue(list);
            }

            @Override
            public void onError(String err) {
                error.setValue(err);
            }
        });
    }
}
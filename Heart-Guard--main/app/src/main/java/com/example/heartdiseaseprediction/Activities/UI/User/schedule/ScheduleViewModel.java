package com.example.heartdiseaseprediction.Activities.UI.User.schedule;


import androidx.lifecycle.MutableLiveData;
import androidx.lifecycle.ViewModel;

import com.example.heartdiseaseprediction.Activities.data.Model.Appointment;
import com.example.heartdiseaseprediction.Activities.data.Repository.ScheduleRepository;

import java.util.List;

public class ScheduleViewModel extends ViewModel {

    private MutableLiveData<List<String>> doctorList = new MutableLiveData<>();
    private MutableLiveData<Boolean> createSuccess = new MutableLiveData<>();
    private MutableLiveData<String> error = new MutableLiveData<>();

    private ScheduleRepository repo = new ScheduleRepository();

    public MutableLiveData<List<String>> getDoctorList() {
        return doctorList;
    }

    public MutableLiveData<Boolean> getCreateSuccess() {
        return createSuccess;
    }

    public MutableLiveData<String> getError() {
        return error;
    }

    // load doctor
    public void loadDoctors(String service) {
        repo.getDoctors(service, new ScheduleRepository.DoctorCallback() {
            @Override
            public void onSuccess(List<String> list) {
                doctorList.setValue(list);
            }

            @Override
            public void onError(String err) {
                error.setValue(err);
            }
        });
    }

    // create appointment
    public void createAppointment(String id, Appointment appointment) {
        repo.createAppointment(id, appointment, new ScheduleRepository.CreateCallback() {
            @Override
            public void onSuccess() {
                createSuccess.setValue(true);
            }

            @Override
            public void onError(String err) {
                error.setValue(err);
            }
        });
    }
}

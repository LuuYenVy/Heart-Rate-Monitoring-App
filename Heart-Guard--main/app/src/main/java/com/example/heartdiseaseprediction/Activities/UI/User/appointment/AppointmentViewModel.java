package com.example.heartdiseaseprediction.Activities.UI.User.appointment;

import android.view.ViewGroup;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;
import androidx.lifecycle.ViewModel;

import com.example.heartdiseaseprediction.Activities.data.Model.Appointment;
import com.example.heartdiseaseprediction.Activities.data.Repository.AppointmentRepository;

import java.util.List;

import android.util.Log;
public class AppointmentViewModel extends ViewModel {
    private MutableLiveData<List<Appointment>> history = new MutableLiveData<>();
    private MutableLiveData<List<Appointment>> incoming = new MutableLiveData<>();

    private MutableLiveData<String> selectedService = new MutableLiveData<>();


    private MutableLiveData<List<String>> services = new MutableLiveData<>();


    public MutableLiveData<List<String>> getServices() {
        return services;
    }

    public MutableLiveData<String> getSelectedService() {
        return selectedService;
    }


    public MutableLiveData<Appointment> getAppointmentLiveData() {
        return appointmentLiveData;
    }

    private MutableLiveData<Appointment> appointmentLiveData= new MutableLiveData<>();

    public MutableLiveData<String> getErrorLiveData() {
        return errorLiveData;
    }

    private MutableLiveData<String> errorLiveData = new MutableLiveData<>();
    private AppointmentRepository repo = new AppointmentRepository();
    public MutableLiveData<List<Appointment>> getHistory(){
        return history;
    }
    public MutableLiveData<List<Appointment>> getIncoming(){
        return incoming;
    }
    public void loadData(String uid){
        repo.getAppointments(uid, "finish", new AppointmentRepository.Callback() {
            @Override
            public void onSuccess(List<Appointment> list) {
                history.postValue(list);
            }

            @Override
            public void onError(String error) {
                Log.e("loadData", error);
            }
        });
        repo.getAppointments(uid, "incoming", new AppointmentRepository.Callback() {
            @Override
            public void onSuccess(List<Appointment> list) {
                incoming.postValue(list);
            }

            @Override
            public void onError(String error) {
                Log.e("getAppointments", error);
            }
        });
    }
    public void loadAppointment(String id){
        repo.getAppointmentById(id, new AppointmentRepository.DetailAppointmentCallback() {
            @Override
            public void onSuccess(Appointment appointment) {
                appointmentLiveData.setValue(appointment);
            }

            @Override
            public void onError(String message) {
                errorLiveData.setValue(message);
            }
        });
    }

    public void loadServices() {
        repo.getServices(new AppointmentRepository.ServiceCallback() {
            @Override
            public void onSuccess(List<String> list) {
                services.setValue(list);
            }

            @Override
            public void onError(String error) {
                Log.e("LoadServicesError", error);
            }
        });
    }

    public void selectService(String service) {
        selectedService.setValue(service);
    }

}

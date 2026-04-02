package com.example.heartdiseaseprediction.Activities.UI.Doctor.examination;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;
import androidx.lifecycle.ViewModel;

import com.example.heartdiseaseprediction.Activities.data.Model.Appointment;
import com.example.heartdiseaseprediction.Activities.data.Repository.AdminAppointmentRepository;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;

import java.util.HashMap;
import java.util.Map;

public class ExaminationViewModel extends ViewModel {

    private MutableLiveData<Boolean> success = new MutableLiveData<>();
    private MutableLiveData<String> error = new MutableLiveData<>();
    private MutableLiveData<Appointment> appointment = new MutableLiveData<>();
    private MutableLiveData<String> errorAp = new MutableLiveData<>();

    private AdminAppointmentRepository repo = new AdminAppointmentRepository();

    public LiveData<Appointment> getAppointment() {
        return appointment;
    }

    public LiveData<String> getErrorAp() {
        return errorAp;
    }

    private final MutableLiveData<String> temperature = new MutableLiveData<>();


    private final MutableLiveData<String> bloodPressure = new MutableLiveData<>();

    private final MutableLiveData<Boolean> isValid = new MutableLiveData<>();
    public MutableLiveData<Boolean> getSuccess() {
        return success;
    }

    public MutableLiveData<String> getError() {
        return error;
    }
    public MutableLiveData<String> getTemperature() {
        return temperature;
    }
    public MutableLiveData<String> getBloodPressure() {
        return bloodPressure;
    }
    public MutableLiveData<Boolean> getIsValid() {
        return isValid;
    }


    // validate input
    public boolean validate(String symptom, String history, String diagnostic) {
        if (symptom == null || symptom.trim().isEmpty()) {
            error.setValue("Please enter symptom");
            return false;
        }
        if (history == null || history.trim().isEmpty()) {
            error.setValue("Please enter medical history");
            return false;
        }
        if (diagnostic == null || diagnostic.trim().isEmpty()) {
            error.setValue("Please enter diagnostic");
            return false;
        }
        return true;
    }


    public void updateAppointment(String appointmentId,
                                  String temperature,
                                  String bloodPressure,
                                  String symptom,
                                  String history,
                                  String diagnostic) {

        DatabaseReference ref = FirebaseDatabase.getInstance()
                .getReference("appointments")
                .child(appointmentId);

        Map<String, Object> data = new HashMap<>();
        data.put("temperater", temperature);
        data.put("blood_pressure", bloodPressure);
        data.put("Sypptom", symptom);
        data.put("MedicalHistory", history);
        data.put("Diagnostic", diagnostic);
        data.put("status", "finish");

        ref.updateChildren(data)
                .addOnSuccessListener(unused -> success.setValue(true))
                .addOnFailureListener(e -> error.setValue(e.getMessage()));
    }
    public void setPreCheckData(String temp, String bp) {
        temperature.setValue(temp);
        bloodPressure.setValue(bp);
    }

    public void validate(String temp, String bp) {
        if (temp == null || temp.isEmpty()) {
            isValid.setValue(false);
        } else if (bp == null || bp.isEmpty()) {
            isValid.setValue(false);
        } else {
            isValid.setValue(true);
        }
    }
    public void loadAppointment(String appointmentId) {

        repo.getAppointmentById(appointmentId, new AdminAppointmentRepository.SingleAppointmentCallback() {
            @Override
            public void onSuccess(Appointment ap) {
                appointment.setValue(ap);
            }

            @Override
            public void onError(String err) {
                error.setValue(err);
            }
        });
    }
}
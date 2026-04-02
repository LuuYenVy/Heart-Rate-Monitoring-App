package com.example.heartdiseaseprediction.Activities.UI.Doctor.prescription;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;
import androidx.lifecycle.ViewModel;

import com.example.heartdiseaseprediction.Activities.data.Model.Medicine;
import com.example.heartdiseaseprediction.Activities.data.Repository.PrescriptionRepository;

import java.util.List;

public class PrescriptionViewModel extends ViewModel {

    private final PrescriptionRepository repository = new PrescriptionRepository();

    private final MutableLiveData<List<String>> medicineList = new MutableLiveData<>();
    private final MutableLiveData<Boolean> isSaved = new MutableLiveData<>();

    public LiveData<List<String>> getMedicineList() {
        return medicineList;
    }

    public LiveData<Boolean> getIsSaved() {
        return isSaved;
    }

    public void loadMedicines() {
        repository.getMedicines(list -> {
            medicineList.setValue(list);
        });
    }

    public void saveMedicines(String appointmentId, List<Medicine> list) {
        repository.saveMedicines(appointmentId, list);
        isSaved.setValue(true);
    }
}
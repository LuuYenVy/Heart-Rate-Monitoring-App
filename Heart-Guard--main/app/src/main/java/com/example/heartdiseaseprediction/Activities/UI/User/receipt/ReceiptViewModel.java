package com.example.heartdiseaseprediction.Activities.UI.User.receipt;

import androidx.lifecycle.MutableLiveData;
import androidx.lifecycle.ViewModel;

import com.example.heartdiseaseprediction.Activities.data.Model.Medicine;
import com.example.heartdiseaseprediction.Activities.data.Repository.ReceiptRepository;

import java.util.List;

public class ReceiptViewModel extends ViewModel {

    private MutableLiveData<List<Medicine>> medicineLiveData = new MutableLiveData<>();
    private MutableLiveData<String> errorLiveData = new MutableLiveData<>();

    private ReceiptRepository repo = new ReceiptRepository();

    public MutableLiveData<List<Medicine>> getMedicineLiveData() {
        return medicineLiveData;
    }

    public MutableLiveData<String> getErrorLiveData() {
        return errorLiveData;
    }

    public void loadMedicines(String appointmentId) {
        repo.getMedicines(appointmentId, new ReceiptRepository.Callback() {
            @Override
            public void onSuccess(List<Medicine> list) {
                medicineLiveData.setValue(list);
            }

            @Override
            public void onError(String error) {
                errorLiveData.setValue(error);
            }
        });
    }
}

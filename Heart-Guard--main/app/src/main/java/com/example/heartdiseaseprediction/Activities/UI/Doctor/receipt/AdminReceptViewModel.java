package com.example.heartdiseaseprediction.Activities.UI.Doctor.receipt;

import androidx.lifecycle.MutableLiveData;
import androidx.lifecycle.ViewModel;

import com.example.heartdiseaseprediction.Activities.data.Model.Medicine;
import com.google.firebase.database.*;

import java.util.ArrayList;
import java.util.List;

public class AdminReceptViewModel extends ViewModel {

    private MutableLiveData<List<Medicine>> medicineList = new MutableLiveData<>();
    private MutableLiveData<String> error = new MutableLiveData<>();

    public MutableLiveData<List<Medicine>> getMedicineList() {
        return medicineList;
    }

    public MutableLiveData<String> getError() {
        return error;
    }

    public void loadMedicines(String appointmentId) {
        DatabaseReference ref = FirebaseDatabase.getInstance()
                .getReference("appointments")
                .child(appointmentId)
                .child("medicines");

        ref.addListenerForSingleValueEvent(new ValueEventListener() {
            @Override
            public void onDataChange(DataSnapshot snapshot) {
                List<Medicine> list = new ArrayList<>();

                for (DataSnapshot item : snapshot.getChildren()) {
                    Medicine m = item.getValue(Medicine.class);
                    if (m != null) list.add(m);
                }

                medicineList.setValue(list);
            }

            @Override
            public void onCancelled(DatabaseError errorDb) {
                error.setValue(errorDb.getMessage());
            }
        });
    }
}

package com.example.heartdiseaseprediction.Activities.UI.User.appointment;

import androidx.cardview.widget.CardView;
import androidx.fragment.app.Fragment;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.os.Bundle;
import androidx.lifecycle.ViewModelProvider;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import com.example.heartdiseaseprediction.Activities.data.Model.Appointment;
import com.example.heartdiseaseprediction.R;
import com.google.firebase.auth.FirebaseAuth;
import com.google.firebase.auth.FirebaseUser;

import java.util.List;

public class AppointmentFragment extends Fragment {

    private CardView layoutHistory, layoutIncoming;
    ImageView btnMakeAppointment;
    private AppointmentViewModel viewModel;
    private String userId;

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container, Bundle savedInstanceState) {

        View view = inflater.inflate(R.layout.fragment_appointment, container, false);
        layoutHistory = view.findViewById(R.id.History);
        layoutIncoming = view.findViewById(R.id.upcoming);
        ImageView btnMakeAppointment = view.findViewById(R.id.makeApointment);

        btnMakeAppointment.setOnClickListener(v -> {
            startActivity(new Intent(getActivity(), ChooseServiceActivity.class));
        });

        FirebaseUser user = FirebaseAuth.getInstance().getCurrentUser();
        if (user != null) {
            userId = user.getUid();
        } else {
            Toast.makeText(getContext(), "User not logged in", Toast.LENGTH_SHORT).show();
            return view;
        }

        // ViewModel
        viewModel = new ViewModelProvider(this).get(AppointmentViewModel.class);

        observeData();
        viewModel.loadData(userId);

        return view;
    }

    // ================== OBSERVE ==================
    private void observeData() {

        viewModel.getHistory().observe(getViewLifecycleOwner(), list -> {
            renderHistory(list);
        });

        viewModel.getIncoming().observe(getViewLifecycleOwner(), list -> {
            renderIncoming(list);
        });
    }

    // ================== RENDER HISTORY ==================
    private void renderHistory(List<Appointment> list) {

        layoutHistory.removeAllViews();

        for (Appointment item : list) {

            View v = LayoutInflater.from(getContext())
                    .inflate(R.layout.item_appointment_history, layoutHistory, false);

            TextView tvService = v.findViewById(R.id.servicename);
            TextView tvDate = v.findViewById(R.id.date);

            tvService.setText(item.getService());
            tvDate.setText(item.getDate());

            v.setOnClickListener(view -> goToDetail(item));

            layoutHistory.addView(v);
        }
    }

    // ================== RENDER INCOMING ==================
    private void renderIncoming(List<Appointment> list) {

        layoutIncoming.removeAllViews();

        for (Appointment item : list) {

            View v = LayoutInflater.from(getContext())
                    .inflate(R.layout.item_appointment_incoming, layoutIncoming, false);

            TextView tvService = v.findViewById(R.id.Serive);
            TextView tvDate = v.findViewById(R.id.dateservice);

            tvService.setText(item.getService());
            tvDate.setText(item.getDate());

            v.setOnClickListener(view -> goToDetail(item));

            layoutIncoming.addView(v);
        }
    }

    // ================== NAVIGATION ==================
    private void goToDetail(Appointment item) {

        SharedPreferences sp = requireContext()
                .getSharedPreferences("MyPrefs", Context.MODE_PRIVATE);

        sp.edit()
                .putString("AppointmentKey", item.getId())
                .putString("DateApoimentSelect", item.getDate())
                .putString("NameApoimentSelect", item.getService())
                .apply();

        startActivity(new Intent(getActivity(), DetailAppointmentActivity.class));
    }
}

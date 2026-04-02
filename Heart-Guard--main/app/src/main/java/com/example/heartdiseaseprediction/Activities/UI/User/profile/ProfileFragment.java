package com.example.heartdiseaseprediction.Activities.UI.User.profile;
import android.app.Dialog;
import android.content.Intent;
import android.os.Bundle;

import androidx.fragment.app.Fragment;
import androidx.lifecycle.ViewModelProvider;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.example.heartdiseaseprediction.Activities.UI.Auth.LoginActivity;
import com.example.heartdiseaseprediction.R;
import com.google.android.material.textfield.TextInputEditText;
import com.google.firebase.auth.FirebaseAuth;

public class ProfileFragment extends Fragment {

    private TextView username, age, height, weight, gender;
    private ImageView logOut, updateInfo;

    private ProfileViewModel viewModel;

    private String userId = FirebaseAuth.getInstance().getCurrentUser().getUid();

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container, Bundle savedInstanceState) {

        View view = inflater.inflate(R.layout.fragment_profile, container, false);

        username = view.findViewById(R.id.textView_Username);
        age = view.findViewById(R.id.age_value);
        height = view.findViewById(R.id.height_value);
        weight = view.findViewById(R.id.weight_value);
        gender = view.findViewById(R.id.sex_value);

        logOut = view.findViewById(R.id.LogOutBtn);
        updateInfo = view.findViewById(R.id.updateInfo);

        viewModel = new ViewModelProvider(this).get(ProfileViewModel.class);

        observeData();
        viewModel.loadUser(userId);

        logOut.setOnClickListener(v -> {
            FirebaseAuth.getInstance().signOut();
            startActivity(new Intent(getActivity(), LoginActivity.class));
            getActivity().finish();
            Toast.makeText(getActivity(), "Logout Successfully", Toast.LENGTH_SHORT).show();
        });

        updateInfo.setOnClickListener(v -> showDialog());

        return view;
    }

    private void observeData() {
        viewModel.getUser().observe(getViewLifecycleOwner(), user -> {
            if (user != null) {
                username.setText(user.getUsername());
                age.setText(user.getAge());
                height.setText(user.getHeight());
                weight.setText(user.getWeight());
                gender.setText(user.getGender());
            }
        });

        viewModel.getMessage().observe(getViewLifecycleOwner(), msg -> {
            Toast.makeText(getContext(), msg, Toast.LENGTH_SHORT).show();
        });
    }

    private void showDialog() {
        Dialog dialog = new Dialog(getActivity());
        dialog.setContentView(R.layout.dialog_options);

        TextInputEditText edtAge = dialog.findViewById(R.id.age_value);
        TextInputEditText edtWeight = dialog.findViewById(R.id.weight_value);
        TextInputEditText edtHeight = dialog.findViewById(R.id.height_value);

        Button btnUpdate = dialog.findViewById(R.id.btn_update_information);

        btnUpdate.setOnClickListener(v -> {
            String a = edtAge.getText().toString();
            String w = edtWeight.getText().toString();
            String h = edtHeight.getText().toString();

            viewModel.updateUser(userId, a, h, w);
            dialog.dismiss();
        });

        dialog.show();
    }
}

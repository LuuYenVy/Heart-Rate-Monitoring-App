package com.example.heartdiseaseprediction.Activities.UI.User;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;

import com.example.heartdiseaseprediction.Activities.UI.User.chatbot.ChatFragment;
import com.example.heartdiseaseprediction.R;
import com.example.heartdiseaseprediction.databinding.ActivityMainScreenBinding;
import com.google.android.material.bottomnavigation.BottomNavigationView;

import androidx.fragment.app.Fragment;
import androidx.fragment.app.FragmentManager;
import androidx.fragment.app.FragmentTransaction;
import com.example.heartdiseaseprediction.Activities.UI.User.home.HomeFragment;
import com.example.heartdiseaseprediction.Activities.UI.User.chart.ChartFragment;
import com.example.heartdiseaseprediction.Activities.UI.User.appointment.AppointmentFragment;
import com.example.heartdiseaseprediction.Activities.UI.User.profile.ProfileFragment;
import com.example.heartdiseaseprediction.R;
public class MainActivity extends AppCompatActivity {

        private ActivityMainScreenBinding binding;

        @Override
        protected void onCreate(Bundle savedInstanceState) {
            super.onCreate(savedInstanceState);

            binding = ActivityMainScreenBinding.inflate(getLayoutInflater());
            setContentView(binding.getRoot());

            setupBottomNav();
            loadDefaultFragment();
        }

        private void setupBottomNav() {
            binding.bottomNavigationView.setSelectedItemId(R.id.home);

            binding.bottomNavigationView.setOnItemSelectedListener(item -> {
                int id= item.getItemId();
                if( id == R.id.home){
                    replaceFragment(new HomeFragment());
                }
                else if( id == R.id.chart){
                    replaceFragment(new ChartFragment());
                }
                else if( id == R.id.history){
                    replaceFragment(new AppointmentFragment());
                }
                else {
                    replaceFragment(new ChatFragment());
                }
                return true;
            });
        }

        private void loadDefaultFragment() {
            replaceFragment(new HomeFragment());
        }

        private void replaceFragment(Fragment fragment) {
            getSupportFragmentManager()
                    .beginTransaction()
                    .replace(R.id.flFragment, fragment)
                    .commit();
        }
    }

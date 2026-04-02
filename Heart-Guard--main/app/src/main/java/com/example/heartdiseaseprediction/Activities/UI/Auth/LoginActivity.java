package com.example.heartdiseaseprediction.Activities.UI.Auth;

import androidx.appcompat.app.AppCompatActivity;
import android.app.ProgressDialog;
import android.content.Intent;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.text.TextUtils;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import androidx.lifecycle.ViewModelProvider;

import com.example.heartdiseaseprediction.Activities.UI.Doctor.main.AdminMainActivity;
import com.example.heartdiseaseprediction.Activities.UI.User.MainActivity;
import com.example.heartdiseaseprediction.R;
import com.example.heartdiseaseprediction.SesionManager.UserSessionManager;
import com.google.android.material.textfield.TextInputEditText;

public class LoginActivity extends AppCompatActivity {
    private TextInputEditText email;
    private TextInputEditText password;
    private Button login, forgotPassword;
    private TextView register;
    private AuthViewModel viewModel;
    private ProgressDialog loadingBar;
    @Override
    protected void onCreate( Bundle saveInstanceState){
        super.onCreate(saveInstanceState);
        setContentView(R.layout.activity_login);
        initializeUI();
        viewModel = new ViewModelProvider(this).get(AuthViewModel.class);
        observeViewModel();
        login.setOnClickListener(v ->handleLogin());
        register.setOnClickListener(v -> {
            startActivity(new Intent(this, RegisterActivity.class));
        });

    }
    private void initializeUI(){
        email = findViewById(R.id.txtUserNameLogin);
        password = findViewById(R.id.passPassLogin);
        login = findViewById(R.id.ButtonLogin);
        register = findViewById(R.id.ButtonSignup);

        loadingBar = new ProgressDialog(this);
    }
    private void handleLogin(){
        String username= email.getText().toString().trim();
        String pwd= password.getText().toString().trim();
        if (TextUtils.isEmpty(username)||TextUtils.isEmpty(pwd)){
            Toast.makeText(this,"Missing input", Toast.LENGTH_SHORT).show();
            return;
        }
        loadingBar.show();
        viewModel.loginWithEmail(username,pwd);
    }

    private void observeViewModel(){
        viewModel.getLoginResult().observe(this, result ->{
            loadingBar.dismiss();
            if(result.isSuccess()){
                if(result.getRole().equals("doctor")){
                    UserSessionManager session = new UserSessionManager(this);
                    session.createDoctorLoginSession(
                            result.getUser().getEmail(),
                            result.getUser().getUsername());
                    startActivity(new Intent(this, AdminMainActivity.class));
                }
                else{
                    UserSessionManager session = new UserSessionManager(this);

                    session.createUserLoginSession(
                            result.getUser().getUserID(),
                            result.getUser().getEmail(),
                            result.getUser().getUsername(),
                            result.getUser().getAge(),
                            result.getUser().getHeight(),
                            result.getUser().getWeight(),
                            result.getUser().getGender());

                    startActivity(new Intent(this, MainActivity.class));
                }
                finish();
            }
            else{
                Toast.makeText(this,result.getMessage(),Toast.LENGTH_SHORT).show();
            }
        });
    }
}

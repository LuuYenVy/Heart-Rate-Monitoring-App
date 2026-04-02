package com.example.heartdiseaseprediction.Activities.UI.Auth;

import android.app.ProgressDialog;
import android.content.Intent;
import android.os.Bundle;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
import androidx.lifecycle.ViewModelProvider;

import com.example.heartdiseaseprediction.R;

public class RegisterActivity extends AppCompatActivity {
    private EditText email, password, confirm, username;
    private Button register;
    private TextView login;
    private ProgressDialog loadingBar;

    private AuthViewModel viewModel;
    @Override
    protected  void onCreate(Bundle saveInstanceState){
        super.onCreate(saveInstanceState); // 🔥 BẮT BUỘC
        setContentView(R.layout.activity_signup);

        initUI();
        initViewModel();
        observeViewModel();

        login.setOnClickListener(v -> finish());
    }
    public void initUI(){
        email=findViewById(R.id.emailEditText);
        password=findViewById(R.id.passEditText);
        confirm=findViewById(R.id.repassEditText);
        username=findViewById(R.id.usernameEdittext);
        register=findViewById(R.id.ButtonSignup);
        login=findViewById(R.id.ButtonLogin);
        loadingBar = new ProgressDialog(this);
        register.setOnClickListener( v -> handleRegister());
    }
    public void initViewModel(){
        viewModel = new ViewModelProvider(this).get(AuthViewModel.class);
    }
    public void handleRegister(){
        loadingBar.setMessage("Creating Account...");
        loadingBar.show();

        if (email == null || password == null || confirm == null || username == null) {
            Toast.makeText(this, "UI error (null view)", Toast.LENGTH_SHORT).show();
            return;
        }

        String emailStr = email.getText() != null ? email.getText().toString().trim() : "";
        String passStr = password.getText() != null ? password.getText().toString().trim() : "";
        String confirmStr = confirm.getText() != null ? confirm.getText().toString().trim() : "";
        String usernameStr = username.getText() != null ? username.getText().toString().trim() : "";

        if (emailStr.isEmpty() || passStr.isEmpty() || confirmStr.isEmpty() || usernameStr.isEmpty()) {
            Toast.makeText(this, "Please fill all fields", Toast.LENGTH_SHORT).show();
            return;
        }

        if (!passStr.equals(confirmStr)) {
            Toast.makeText(this, "Password not match", Toast.LENGTH_SHORT).show();
            return;
        }

        loadingBar.setMessage("Creating Account...");
        loadingBar.setCanceledOnTouchOutside(false);
        loadingBar.show();

        viewModel.register(emailStr, passStr, confirmStr, usernameStr);

    }
    private void observeViewModel(){
        viewModel.getregisterResult().observe(this, result -> {
            loadingBar.dismiss();
            if(result.isSuccess()){
                startActivity(new Intent(this, SignUpInformationActivity.class));
            }
            else{
                Toast.makeText(this,result.getMessage(),Toast.LENGTH_SHORT).show();
            }
        });
    }

}

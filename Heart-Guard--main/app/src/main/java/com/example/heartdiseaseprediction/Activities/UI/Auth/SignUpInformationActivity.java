package com.example.heartdiseaseprediction.Activities.UI.Auth;

import android.content.Intent;
import android.os.Bundle;
import android.widget.ArrayAdapter;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.Spinner;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
import androidx.lifecycle.ViewModel;
import androidx.lifecycle.ViewModelProvider;

import com.example.heartdiseaseprediction.R;

public class SignUpInformationActivity extends AppCompatActivity {
    private Spinner gender;
    private EditText weight, height, age;
    private ImageView finish;
    private AuthViewModel viewModel;

    @Override
    protected void onCreate(Bundle savedInstanceState){
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_signup_information);

        initUI();
        viewModel = new ViewModelProvider(this).get(AuthViewModel.class);
        observeViewModel();
        finish.setOnClickListener(v-> handlesubmit());
    }
    private void initUI(){
        gender=findViewById(R.id.gender);
        weight=findViewById(R.id.weight);
        age=findViewById(R.id.age);
        height=findViewById(R.id.height);
        finish=findViewById(R.id.finish);

        String[] services = {"Male","Female"};
        ArrayAdapter<String> adapter = new ArrayAdapter<>(this,
                android.R.layout.simple_spinner_item,services);
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        gender.setAdapter(adapter);
    }
    private void handlesubmit(){
        String g=gender.getSelectedItem().toString();
        String w=weight.getText().toString();
        String h= height.getText().toString();
        String a= age.getText().toString();
        viewModel.saveUserInfo(g,w,h,a);
    }
    private void observeViewModel(){
        viewModel.getSaveInfoResult().observe(this,result ->{
            if(result.isSuccess()){
                Toast.makeText(this,"Sign Up Successfully!",Toast.LENGTH_SHORT).show();
                startActivity(new Intent(this, LoginActivity.class));
                finish();
            }
            else{
                Toast.makeText(this,result.getMessage(),Toast.LENGTH_SHORT).show();
            }
        });
    }
}

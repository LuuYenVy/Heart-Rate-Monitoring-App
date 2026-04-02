package com.example.heartdiseaseprediction.Activities.UI.User.chatbot;

import android.os.Bundle;
import android.text.TextUtils;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;

import androidx.fragment.app.Fragment;
import androidx.lifecycle.ViewModelProvider;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import android.widget.EditText;

import com.example.heartdiseaseprediction.R;
import com.example.heartdiseaseprediction.Activities.data.Model.Message;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class ChatFragment extends Fragment {

    private ChatViewModel viewModel;

    private RecyclerView recyclerView;
    private Adapter adapter;
    private List<Message> messages;

    private EditText input;
    private Button sendButton;

    public ChatFragment() {
    }

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {

        View view = inflater.inflate(R.layout.fragment_chat, container, false);

        recyclerView = view.findViewById(R.id.recyclerView);
        input = view.findViewById(R.id.input);
        sendButton = view.findViewById(R.id.sendButton);

        messages = new ArrayList<>();
        adapter = new Adapter(messages);

        recyclerView.setLayoutManager(new LinearLayoutManager(getContext()));
        recyclerView.setAdapter(adapter);

        viewModel = new ViewModelProvider(this).get(ChatViewModel.class);

        // observe AI response
        viewModel.getResponse().observe(getViewLifecycleOwner(), result -> {
            messages.add(new Message(result, false)); // bot
            adapter.notifyDataSetChanged();
            recyclerView.scrollToPosition(messages.size() - 1);
        });

        sendButton.setOnClickListener(v -> {

            String question = input.getText().toString();

            if (TextUtils.isEmpty(question)) return;

            messages.add(new Message(question, true));
            adapter.notifyDataSetChanged();

            List<Integer> heartRates = Arrays.asList(72, 80, 95, 110);

            viewModel.sendToAI(heartRates, question);

            input.setText("");
        });

        return view;
    }
}
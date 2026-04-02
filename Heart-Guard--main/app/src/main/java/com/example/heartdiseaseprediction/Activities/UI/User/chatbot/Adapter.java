package com.example.heartdiseaseprediction.Activities.UI.User.chatbot;

import android.view.Gravity;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.LinearLayout;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import com.example.heartdiseaseprediction.R;
import com.example.heartdiseaseprediction.Activities.data.Model.Message;
import java.util.List;

public class Adapter extends RecyclerView.Adapter<Adapter.ViewHolder> {

    private List<Message> messages;

    public Adapter(List<Message> messages) {
        this.messages = messages;
    }

    public static class ViewHolder extends RecyclerView.ViewHolder {
        TextView text;
        LinearLayout container;

        public ViewHolder(View view) {
            super(view);
            text = view.findViewById(R.id.messageText);
            container = view.findViewById(R.id.container);
        }
    }

    @NonNull
    @Override
    public ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View v = LayoutInflater.from(parent.getContext())
                .inflate(R.layout.item_message, parent, false);
        return new ViewHolder(v);
    }

    @Override
    public void onBindViewHolder(@NonNull ViewHolder holder, int position) {
        Message msg = messages.get(position);

        holder.text.setText(msg.getText());

        LinearLayout.LayoutParams params =
                (LinearLayout.LayoutParams) holder.text.getLayoutParams();

        if (msg.isUser()) {
            params.gravity = Gravity.END;
            holder.text.setBackgroundResource(R.drawable.bg_chat_user);
        } else {
            params.gravity = Gravity.START;
            holder.text.setBackgroundResource(R.drawable.bg_chat_bot);
        }

        holder.text.setLayoutParams(params);
    }

    @Override
    public int getItemCount() {
        return messages.size();
    }
}
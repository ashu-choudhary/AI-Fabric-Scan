package com.example.sareeclassifier

import android.content.Intent
import android.os.Bundle
import android.widget.Button
import android.widget.ImageButton
import androidx.appcompat.app.AppCompatActivity

class WelcomeActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        setTheme(R.style.Theme_SareeClassifier) // Apply dark/light theme on startup
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_welcome)

        val startButton = findViewById<ImageButton>(R.id.startButton) // ✅ Correct cast

        startButton.setOnClickListener {
            val intent = Intent(this, MainActivity::class.java)
            startActivity(intent)
            finish()
        }
    }
}

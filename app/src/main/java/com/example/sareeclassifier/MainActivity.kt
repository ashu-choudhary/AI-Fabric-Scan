package com.example.sareeclassifier

import android.annotation.SuppressLint
import android.app.AlertDialog
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.PorterDuff
import android.graphics.PorterDuffColorFilter
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.graphics.createBitmap
import com.yalantis.ucrop.UCrop
import org.tensorflow.lite.Interpreter
import yuku.ambilwarna.AmbilWarnaDialog
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.UUID

@Suppress("DEPRECATION")
class MainActivity : AppCompatActivity() {

    private lateinit var imageView: ImageView
    private lateinit var resultTextView: TextView
    private lateinit var cropButton: Button
    private lateinit var recolorButton: Button
    private lateinit var undoButton: Button
    private lateinit var tflite: Interpreter

    private var originalBitmap: Bitmap? = null
    private var currentBitmap: Bitmap? = null
    private var previousBitmap: Bitmap? = null

    private val cameraLauncher = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
        if (result.resultCode == RESULT_OK) {
            val bitmap = result.data?.extras?.get("data") as Bitmap
            setImage(bitmap)
        }
    }

    private val galleryLauncher = registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
        uri?.let {
            val inputStream = contentResolver.openInputStream(uri)
            val bitmap = BitmapFactory.decodeStream(inputStream)
            setImage(bitmap)
        }
    }

    private fun setImage(bitmap: Bitmap) {
        originalBitmap = bitmap
        currentBitmap = bitmap.copy(bitmap.config!!, true)
        imageView.setImageBitmap(currentBitmap)
        classifyImage(currentBitmap!!)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        imageView = findViewById(R.id.imageView)
        resultTextView = findViewById(R.id.resultTextView)
        val selectButton: Button = findViewById(R.id.selectImageButton)
        cropButton = findViewById(R.id.cropImageButton)
        recolorButton = findViewById(R.id.btnRecolorImage)
        undoButton = findViewById(R.id.btnUndoRecolor)

        tflite = Interpreter(loadModelFile())

        selectButton.setOnClickListener { showImageSourceDialog() }
        cropButton.setOnClickListener { launchCropper() }

        recolorButton.setOnClickListener {
            showColorPickerDialog()
        }

        undoButton.setOnClickListener {
            if (previousBitmap != null) {
                currentBitmap = previousBitmap
                imageView.setImageBitmap(currentBitmap)
                classifyImage(currentBitmap!!)
            } else {
                Toast.makeText(this, "Nothing to undo", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun showImageSourceDialog() {
        val options = arrayOf("Camera", "Gallery")
        AlertDialog.Builder(this)
            .setTitle("Choose Image Source")
            .setItems(options) { _, which ->
                when (which) {
                    0 -> openCamera()
                    1 -> openGallery()
                }
            }
            .show()
    }

    private fun openCamera() {
        val intent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
        cameraLauncher.launch(intent)
    }

    private fun openGallery() {
        galleryLauncher.launch("image/*")
    }

    private fun loadModelFile(): ByteBuffer {
        val fileDescriptor = assets.openFd("saree_classifier.tflite")
        val inputStream = fileDescriptor.createInputStream()
        val byteArray = ByteArray(fileDescriptor.length.toInt())
        inputStream.read(byteArray)
        val byteBuffer = ByteBuffer.allocateDirect(byteArray.size)
        byteBuffer.order(ByteOrder.nativeOrder())
        byteBuffer.put(byteArray)
        return byteBuffer
    }

    @SuppressLint("UseKtx", "SetTextI18n")
    private fun classifyImage(bitmap: Bitmap) {
        val resized = Bitmap.createScaledBitmap(bitmap, 128, 128, true)
        val inputBuffer = ByteBuffer.allocateDirect(128 * 128 * 3 * 4).order(ByteOrder.nativeOrder())
        val intValues = IntArray(128 * 128)
        resized.getPixels(intValues, 0, 128, 0, 0, 128, 128)

        for (pixel in intValues) {
            inputBuffer.putFloat((pixel shr 16 and 0xFF) / 255f)
            inputBuffer.putFloat((pixel shr 8 and 0xFF) / 255f)
            inputBuffer.putFloat((pixel and 0xFF) / 255f)
        }

        val output = Array(1) { FloatArray(2) }
        tflite.run(inputBuffer, output)

        val label = if (output[0][0] > output[0][1]) "Handloom Saree" else "Powerloom Saree"
        resultTextView.text = "Prediction: $label"
    }

    private fun launchCropper() {
        currentBitmap?.let {
            val uri = getImageUriFromBitmap(it)
            val destinationUri = Uri.fromFile(File(cacheDir, UUID.randomUUID().toString() + ".jpg"))
            UCrop.of(uri, destinationUri).start(this)
        } ?: Toast.makeText(this, "No image to crop", Toast.LENGTH_SHORT).show()
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        if (resultCode == RESULT_OK && requestCode == UCrop.REQUEST_CROP) {
            data?.let {
                val resultUri = UCrop.getOutput(it)
                resultUri?.let { uri ->
                    val bitmap = BitmapFactory.decodeStream(contentResolver.openInputStream(uri))
                    setImage(bitmap)
                }
            }
        }
    }

    private fun getImageUriFromBitmap(bitmap: Bitmap): Uri {
        val file = File(cacheDir, "temp.jpg")
        file.outputStream().use {
            bitmap.compress(Bitmap.CompressFormat.JPEG, 100, it)
        }
        return Uri.fromFile(file)
    }

    private fun showColorPickerDialog() {
        val dialog = AmbilWarnaDialog(
            this,
            Color.RED,
            object : AmbilWarnaDialog.OnAmbilWarnaListener {
                override fun onCancel(dialog: AmbilWarnaDialog?) {}
                override fun onOk(dialog: AmbilWarnaDialog?, color: Int) {
                    currentBitmap?.let { bitmap ->
                        previousBitmap = bitmap.copy(bitmap.config!!, true)
                        val filtered = applyColorFilter(bitmap, color)
                        currentBitmap = filtered
                        imageView.setImageBitmap(filtered)
                        classifyImage(filtered)
                    }
                }
            }
        )
        dialog.show()
    }

    private fun applyColorFilter(bitmap: Bitmap, color: Int): Bitmap {
        val result = createBitmap(bitmap.width, bitmap.height, bitmap.config!!)
        val canvas = Canvas(result)
        val paint = Paint()
        val filter = PorterDuffColorFilter(color, PorterDuff.Mode.MULTIPLY)
        paint.colorFilter = filter
        canvas.drawBitmap(bitmap, 0f, 0f, paint)
        return result
    }
}

package com.example.videoactionclassification

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.ProgressBar
import android.widget.TextView
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import org.tensorflow.lite.support.label.Category

class MainActivity : AppCompatActivity() {
    private val TAG = "POC"

    companion object {
        private const val MAX_RESULT = 3
        private const val MODEL_MOVINET_A2_FILE = "movinet_a2_stream_int8.tflite"
        private const val MODEL_LABEL_FILE = "kinetics600_label_map.txt"
        private const val MODEL_FPS = 5
        private const val REQUEST_CODE = 123
        private const val RESET_AFTER = 15
    }

    private val lock = Any()
    private var videoClassifier: VideoClassifier? = null
    private var numThread = 1
    private lateinit var textViewDetections: TextView
    private lateinit var buttonLoadVideo: Button
    private lateinit var imageViewVideo: ImageView
    private lateinit var progress: ProgressBar
    private val dataRetriever = MediaMetadataRetriever()
    private var rotation = "0"

    private val pickMedia = registerForActivityResult(ActivityResultContracts.PickVisualMedia()) {
        if (it != null) {
            processVideo(it)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        textViewDetections = findViewById(R.id.textview_detections)
        imageViewVideo = findViewById(R.id.imageview)
        buttonLoadVideo = findViewById(R.id.button_load)
        progress = findViewById(R.id.progress_load)
        buttonLoadVideo.setOnClickListener {
            selectVideo()
        }
        createClassifier()
    }

    private fun selectVideo() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            if (checkSelfPermission(Manifest.permission.READ_MEDIA_VIDEO) != PackageManager.PERMISSION_GRANTED) {
                requestPermissions(arrayOf(Manifest.permission.READ_MEDIA_VIDEO), REQUEST_CODE)
                return
            }
        } else {
            if (checkSelfPermission(Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
                requestPermissions(arrayOf(Manifest.permission.READ_EXTERNAL_STORAGE), REQUEST_CODE)
                return
            }
        }
        pickMedia.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.VideoOnly))
    }

    private fun createClassifier() {
        synchronized(lock) {
            if (videoClassifier != null) {
                videoClassifier?.close()
                videoClassifier = null
            }
            val options =
                VideoClassifier.VideoClassifierOptions.builder()
                    .setMaxResult(MAX_RESULT)
                    .setNumThreads(numThread)
                    .build()
            val modelFile = MODEL_MOVINET_A2_FILE
            videoClassifier = VideoClassifier.createFromFileAndLabelsAndOptions(
                this,
                modelFile,
                MODEL_LABEL_FILE,
                options
            )
            Log.d(TAG, "Classifier created.")
        }
    }

    private fun processVideo(uri: Uri) {
        textViewDetections.text = ""

        val videoPath = Utilities.getMediaStorePath(this, uri)
        Log.d(TAG, "processVideo() $videoPath")
        dataRetriever.setDataSource(videoPath)

        val duration = dataRetriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLong() ?: 0L
        rotation = dataRetriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_ROTATION) ?: "0"

        Log.d(TAG, "processVideo() duration : $duration")
        Log.d(TAG, "processVideo() rotation : $rotation")
        Log.d(TAG, "processVideo() frameCount: ")

        progress.max = (duration / 1000).toInt()
        progress.progress = 0

        val frameStep = 1000L / MODEL_FPS
        Log.d(TAG, "processVideo() frameStep : $frameStep")

        videoClassifier?.reset()
        var results: List<Category>? = null

        var resetTime = 0L

        CoroutineScope(Dispatchers.Default).launch {
            for (i in 0 until duration step frameStep) {
                synchronized(lock) {
                    val frame = getVideoFrame(i)
                    val t = i / 1000F
                    progress.progress = t.toInt()

                    resetTime += frameStep

                    if (resetTime > RESET_AFTER * 1000L) {
                        videoClassifier?.reset()
                        Log.d(TAG, "processVideo() reset at : $t")
                        showResults(t, results)
                        resetTime = 0
                    }

                    Log.d(TAG, "processVideo() frame at : $t")

                    frame?.let {
                        results = videoClassifier?.classify(frame)
                        val processedImage = videoClassifier?.preprocessInputImageForView(frame)
                        Handler(Looper.getMainLooper()).post {
                            imageViewVideo.setImageBitmap(processedImage!!.bitmap)
                        }
                    }
                }
            }
            showResults(duration / 1000F, results)
            progress.progress = progress.max
        }
    }

    private fun showResults(t: Float, labels: List<Category>?) {
        runOnUiThread {
            val listStr = labels?.map {
                return@map "Class : " + it.label + ", score : " + it.score
            }?.take(MAX_RESULT)
            listStr?.let {
                textViewDetections.text =  textViewDetections.text.toString() + "\n\nDetections at $t : \n" + it.joinToString("\n")
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        videoClassifier?.close()
        dataRetriever.release()
    }

    private fun getVideoFrame(frameTime: Long): Bitmap? {
        try {
            var frame = dataRetriever.getFrameAtTime(frameTime * 1000)
            if (frame != null) {
                Log.d(TAG, "getVideoFrame() ${frame.width}, ${frame.height}, ${frame.config}")
                if (frame.config != Bitmap.Config.ARGB_8888) {
                    frame = frame.copy(Bitmap.Config.ARGB_8888, true)
                }
                if (rotation == "90") {
                    //frame = frame!!.rotate(90F)
                    return frame
                }
                return frame
            } else {
                Log.d(TAG, "getVideoFrame() failed to get frame at $frameTime")
                return null
            }
        } catch (e: Exception) {
            e.printStackTrace()
            return null
        }
    }
}
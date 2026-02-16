using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using Unity.Sentis;

public class SentisSsdRunner : MonoBehaviour
{
    [Header("Model")]
    public ModelAsset modelAsset;

    [Header("Scene refs")]
    public RawImage video;
    public DetectionOverlay overlay;

    [Header("Inference")]
    public BackendType backend = BackendType.CPU;   // alusta CPU-ga, hiljem GPUCompute
    [Range(0.05f, 0.5f)] public float intervalSec = 0.15f; // 6-7 FPS inference
    [Range(0f, 1f)] public float scoreThreshold = 0.45f;

    Worker _worker;
    WebCamTexture _cam;
    Tensor<float> _input;
    TextureTransform _tx;
    float _nextTime;

    // temp buffers
    readonly List<DetectionOverlay.Detection> _dets = new();

    void Start()
    {
        // 1) Start camera
        _cam = new WebCamTexture(640, 480, 30);
        _cam.Play();
        if (video != null) video.texture = _cam;

        // 2) Load model + worker
        var model = ModelLoader.Load(modelAsset);

        // API variant A (newer Sentis)
        _worker = new Worker(model, backend);

        // 3) Input tensor (NHWC 1x300x300x3 for SSD Mobilenet V1)
        _input = new Tensor<float>(new TensorShape(1, 300, 300, 3));
        _tx = new TextureTransform()
            .SetDimensions(300, 300, 3)
            .SetTensorLayout(TensorLayout.NHWC)
            .SetCoordOrigin(CoordOrigin.BottomLeft);

        _nextTime = Time.time + 1.0f;
    }

    void Update()
    {
        if (_cam == null || !_cam.isPlaying || _cam.width < 16) return;
        if (Time.time < _nextTime) return;
        _nextTime = Time.time + intervalSec;

        RunOnce();
    }

    void RunOnce()
    {
        // Copy camera -> tensor (no alloc)
        TextureConverter.ToTensor(_cam, _input, _tx);

        // Run inference
        _worker.Schedule(_input);

        // Read SSD outputs
        var numT = _worker.PeekOutput("num_detections:0") as Tensor<float>;
        var boxesT = _worker.PeekOutput("detection_boxes:0") as Tensor<float>;
        var scoresT = _worker.PeekOutput("detection_scores:0") as Tensor<float>;
        var classesT = _worker.PeekOutput("detection_classes:0") as Tensor<float>;

        if (numT == null || boxesT == null || scoresT == null || classesT == null)
        {
            Debug.LogError("SSD outputs not found. Check output names in ModelAsset inspector (Sentis).");
            return;
        }

        float[] num = numT.DownloadToArray();
        float[] boxes = boxesT.DownloadToArray();     // [N,4] => top,left,bottom,right
        float[] scores = scoresT.DownloadToArray();   // [N]
        float[] classes = classesT.DownloadToArray(); // [N]

        int n = Mathf.Clamp((int)num[0], 0, scores.Length);

        _dets.Clear();
        for (int i = 0; i < n; i++)
        {
            int bi = i * 4;
            _dets.Add(new DetectionOverlay.Detection
            {
                top = boxes[bi + 0],
                left = boxes[bi + 1],
                bottom = boxes[bi + 2],
                right = boxes[bi + 3],
                score = scores[i],
                classId = (int)classes[i],
            });
        }

        overlay.Render(_dets, scoreThreshold);
    }

    void OnDestroy()
    {
        _worker?.Dispose();
        _input?.Dispose();
        if (_cam != null && _cam.isPlaying) _cam.Stop();
    }
}

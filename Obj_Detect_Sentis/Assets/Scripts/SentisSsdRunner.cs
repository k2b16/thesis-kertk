using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using Unity.Sentis;

public class SentisSsdRunner : MonoBehaviour
{
    public TextAsset labelsTxt;
    string[] _labels;

    [Header("Model")]
    public ModelAsset modelAsset;

    [Tooltip("Output probabilities")]
    public string scoresOutputName = "scores";

    [Tooltip("Output boxes.")]
    public string boxesOutputName = "boxes";

    [Header("Scene refs")]
    public RawImage video;
    public DetectionOverlay overlay;

    [Header("Inference")]
    public BackendType backend = BackendType.CPU;
    [Range(0.05f, 0.5f)] public float intervalSec = 0.2f;
    [Range(0f, 1f)] public float scoreThreshold = 0.5f;

    [Header("NMS")]
    [Range(0.1f, 0.9f)] public float iouThreshold = 0.45f;
    public int candidateSize = 200;
    public int maxDetections = 20;

    [Header("Texture -> Tensor")]
    public CoordOrigin coordOrigin = CoordOrigin.TopLeft;

    Worker _worker;
    WebCamTexture _cam;
    Tensor<float> _input;
    TextureTransform _tx;
    float _nextTime;

    //buffers
    readonly List<DetectionOverlay.Detection> _final = new();
    readonly Dictionary<int, List<DetectionOverlay.Detection>> _perClass = new();

    const int ImageSize = 300;
    const float CenterVariance = 0.1f;
    const float SizeVariance = 0.2f;

    readonly List<Vector4> _priors = new();
    bool _priorsReady = false;

    struct Spec
    {
        public int fmap;
        public int shrink;
        public float boxMin;
        public float boxMax;
        public float[] ratios;
        public Spec(int fmap, int shrink, float boxMin, float boxMax, float[] ratios)
        { this.fmap = fmap; this.shrink = shrink; this.boxMin = boxMin; this.boxMax = boxMax; this.ratios = ratios; }
    }

    static readonly Spec[] Specs = new[]
    {
    new Spec(19, 16,  60, 105, new float[]{2,3}),
    new Spec(10, 32, 105, 150, new float[]{2,3}),
    new Spec(5,  64, 150, 195, new float[]{2,3}),
    new Spec(3, 100, 195, 240, new float[]{2,3}),
    new Spec(2, 150, 240, 285, new float[]{2,3}),
    new Spec(1, 300, 285, 330, new float[]{2,3}),
};

    void BuildPriors()
    {
        _priors.Clear();

        foreach (var spec in Specs)
        {
            float scale = (float)ImageSize / spec.shrink;

            for (int j = 0; j < spec.fmap; j++)
                for (int i = 0; i < spec.fmap; i++)
                {
                    float cx = (i + 0.5f) / scale;
                    float cy = (j + 0.5f) / scale;

                    //small
                    float size = spec.boxMin;
                    float w = size / ImageSize;
                    float h = size / ImageSize;
                    _priors.Add(new Vector4(cx, cy, w, h));

                    //big
                    size = Mathf.Sqrt(spec.boxMin * spec.boxMax);
                    w = size / ImageSize;
                    h = size / ImageSize;
                    _priors.Add(new Vector4(cx, cy, w, h));

                    size = spec.boxMin;
                    w = size / ImageSize;
                    h = size / ImageSize;

                    foreach (var ar in spec.ratios)
                    {
                        float r = Mathf.Sqrt(ar);
                        _priors.Add(new Vector4(cx, cy, w * r, h / r));
                        _priors.Add(new Vector4(cx, cy, w / r, h * r));
                    }
                }
        }

        //clamp
        for (int k = 0; k < _priors.Count; k++)
        {
            var p = _priors[k];
            p.x = Mathf.Clamp01(p.x);
            p.y = Mathf.Clamp01(p.y);
            p.z = Mathf.Clamp01(p.z);
            p.w = Mathf.Clamp01(p.w);
            _priors[k] = p;
        }

        _priorsReady = true;
        Debug.Log("Priors built: " + _priors.Count);
    }

    IEnumerator Start()
    {
        if (labelsTxt != null) _labels = labelsTxt.text.Split(new[] { "\r\n", "\n" }, StringSplitOptions.RemoveEmptyEntries);
        else _labels = null;

        var devices = WebCamTexture.devices;
        Debug.Log("webcams: " + (devices?.Length ?? 0));
        foreach (var d in devices) Debug.Log($"Cam: {d.name}, front={d.isFrontFacing}");

        if (devices == null || devices.Length == 0)
        {
            Debug.LogWarning("No webcam devices found. Disabling SentisSsdRunner.");
            enabled = false;
            yield break;
        }

        //prefer PV cam
        string chosen = devices[0].name;
        foreach (var d in devices) if (!d.isFrontFacing) { chosen = d.name; break; }

        _cam = new WebCamTexture(chosen, 896, 504, 30);
        _cam.Play();
        Debug.Log("Chosen cam: " + chosen);

        if (video != null)
        {
            video.color = Color.white;
            video.material = null;
        }

        while (_cam != null && _cam.isPlaying && _cam.width < 16) yield return null;
        if (video != null) video.texture = _cam;

        var model = ModelLoader.Load(modelAsset);
        _worker = new Worker(model, backend);

        _input = new Tensor<float>(new TensorShape(1, 3, 300, 300));
        _tx = new TextureTransform()
            .SetDimensions(300, 300, 3)
            .SetTensorLayout(TensorLayout.NCHW)
            .SetCoordOrigin(coordOrigin);

        _nextTime = Time.time + 1.0f;
        BuildPriors();
    }

    void Update()
    {
        if (_cam == null) return;
        if (Time.frameCount % 60 == 0) Debug.Log($"Cam playin = {_cam.isPlaying} size={_cam.width}x{_cam.height} updated = {_cam.didUpdateThisFrame}");
        if (_cam == null || !_cam.isPlaying || _cam.width < 16) return;
        if (Time.time < _nextTime) return;
        _nextTime = Time.time + intervalSec;

        RunOnce();
    }

    void RunOnce()
    {
        TextureConverter.ToTensor(_cam, _input, _tx);
        //NormalizeInputInPlace(_input);

        _worker.Schedule(_input);

        var scoresT = _worker.PeekOutput(scoresOutputName) as Tensor<float>;
        var boxesT = _worker.PeekOutput(boxesOutputName) as Tensor<float>;

        if (scoresT == null || boxesT == null)
        {
            Debug.LogError($"Outputs not found");
            return;
        }

        Tensor<float> scoresCPU = scoresT;
        Tensor<float> boxesCPU = boxesT;

        if (backend != BackendType.CPU)
        {
            scoresCPU = scoresT.ReadbackAndClone();
            boxesCPU = boxesT.ReadbackAndClone();
        }

        float[] scores = scoresCPU.DownloadToArray();
        float[] boxes = boxesCPU.DownloadToArray();

        if (backend != BackendType.CPU)
        {
            scoresCPU.Dispose();
            boxesCPU.Dispose();
        }

        DecodeAndNms(scoresT.shape, scores, boxesT.shape, boxes);
        overlay.Render(_final, scoreThreshold);
    }

    void DecodeAndNms(TensorShape scoresShape, float[] scores, TensorShape boxesShape, float[] boxes)
    {
        int numPriors = scoresShape[1];
        int numClasses = scoresShape[2];

        _final.Clear();
        _perClass.Clear();

        for (int i = 0; i < numPriors; i++)
        {
            int bestC = -1;
            float bestS = 0f;

            int sBase = i * numClasses;
            for (int c = 1; c < numClasses; c++)
            {
                float s = scores[sBase + c];
                if (s > bestS) { bestS = s; bestC = c; }
            }

            if (bestC < 0 || bestS < scoreThreshold) continue;

            if (!_priorsReady) BuildPriors();
            if (_priors.Count != numPriors)
            {
                Debug.LogError($"Priors {_priors.Count} != model priors {numPriors}");
                return;
            }

            int bBase = i * 4;

            float dx = boxes[bBase + 0];
            float dy = boxes[bBase + 1];
            float dw = boxes[bBase + 2];
            float dh = boxes[bBase + 3];

            var pr = _priors[i];

            float cx = dx * CenterVariance * pr.z + pr.x;
            float cy = dy * CenterVariance * pr.w + pr.y;
            float w = Mathf.Exp(dw * SizeVariance) * pr.z;
            float h = Mathf.Exp(dh * SizeVariance) * pr.w;

            float xmin = cx - w * 0.5f;
            float ymin = cy - h * 0.5f;
            float xmax = cx + w * 0.5f;
            float ymax = cy + h * 0.5f;

            xmin = Mathf.Clamp01(xmin);
            ymin = Mathf.Clamp01(ymin);
            xmax = Mathf.Clamp01(xmax);
            ymax = Mathf.Clamp01(ymax);

            if (xmax <= xmin || ymax <= ymin) continue;

            var d = new DetectionOverlay.Detection
            {
                left = xmin,
                top = ymin,
                right = xmax,
                bottom = ymax,
                score = bestS,
                classId = bestC,
                label = (_labels != null && bestC < _labels.Length) ? _labels[bestC] : bestC.ToString()
            };

            if (!_perClass.TryGetValue(bestC, out var list))
            {
                list = new List<DetectionOverlay.Detection>();
                _perClass[bestC] = list;
            }
            list.Add(d);
        }

        foreach (var kv in _perClass)
        {
            var kept = HardNms(kv.Value, iouThreshold, candidateSize);
            _final.AddRange(kept);
        }

        _final.Sort((a, b) => b.score.CompareTo(a.score));
        if (_final.Count > maxDetections)
            _final.RemoveRange(maxDetections, _final.Count - maxDetections);
    }

    List<DetectionOverlay.Detection> HardNms(List<DetectionOverlay.Detection> dets, float iouThr, int candSize)
    {
        dets.Sort((a, b) => b.score.CompareTo(a.score));
        if (dets.Count > candSize)
            dets = dets.GetRange(0, candSize);

        var kept = new List<DetectionOverlay.Detection>();
        for (int i = 0; i < dets.Count; i++)
        {
            var d = dets[i];
            bool keep = true;

            for (int k = 0; k < kept.Count; k++)
            {
                if (IoU(d, kept[k]) > iouThr) { keep = false; break; }
            }

            if (keep) kept.Add(d);
        }
        return kept;
    }

    float IoU(DetectionOverlay.Detection a, DetectionOverlay.Detection b)
    {
        float ix1 = Mathf.Max(a.left, b.left);
        float iy1 = Mathf.Max(a.top, b.top);
        float ix2 = Mathf.Min(a.right, b.right);
        float iy2 = Mathf.Min(a.bottom, b.bottom);

        float iw = Mathf.Max(0f, ix2 - ix1);
        float ih = Mathf.Max(0f, iy2 - iy1);
        float inter = iw * ih;

        float areaA = (a.right - a.left) * (a.bottom - a.top);
        float areaB = (b.right - b.left) * (b.bottom - b.top);

        return inter / (areaA + areaB - inter + 1e-5f);
    }

    /*static void NormalizeInputInPlace(Tensor<float> t)
    {
        var span = t.AsSpan();
        for (int i = 0; i < span.Length; i++)
            span[i] = (span[i] * 255f - 127f) / 128f;
    }*/
    void OnDestroy()
    {
        _worker?.Dispose();
        _input?.Dispose();
        if (_cam != null && _cam.isPlaying) _cam.Stop();
    }
}
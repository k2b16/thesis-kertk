using System.Collections.Generic;
using TMPro;
using UnityEngine;

public class DetectionOverlay : MonoBehaviour
{
    [System.Serializable]
    public struct Detection
    {
        //normalized 0..1
        public float left, top, right, bottom;
        public float score;
        public int classId;
    }

    public RectTransform overlayRoot;
    public RectTransform boxPrefab;
    public int maxBoxes = 30;

    readonly List<RectTransform> _pool = new();

    void Awake()
    {
        if (overlayRoot == null) overlayRoot = (RectTransform)transform;
        WarmPool();
    }

    void WarmPool()
    {
        while (_pool.Count < maxBoxes)
        {
            var rt = Instantiate(boxPrefab, overlayRoot);
            rt.gameObject.SetActive(false);
            _pool.Add(rt);
        }
    }

    public void Render(List<Detection> dets, float scoreThreshold)
    {
        WarmPool();

        for (int i = 0; i < _pool.Count; i++)
            _pool[i].gameObject.SetActive(false);

        int shown = 0;
        for (int i = 0; i < dets.Count && shown < _pool.Count; i++)
        {
            var d = dets[i];
            if (d.score < scoreThreshold) continue;

            float l = Mathf.Min(d.left, d.right);
            float r = Mathf.Max(d.left, d.right);
            float t = Mathf.Min(d.top, d.bottom);
            float b = Mathf.Max(d.top, d.bottom);

            //if ((r - 1) < 0.0001f || (b - t) < 0.0001f) continue;
            float xMin = Mathf.Clamp01(l);
            float xMax = Mathf.Clamp01(r);
            float yMin = Mathf.Clamp01(1f - b);
            float yMax = Mathf.Clamp01(1f - t);

            var box = _pool[shown++];
            box.pivot = new Vector2(0.5f, 0.5f);
            box.anchorMin = new Vector2(xMin, yMin);
            box.anchorMax = new Vector2(xMax, yMax);
            box.offsetMin = Vector2.zero;
            box.offsetMax = Vector2.zero;

            var txt = box.GetComponentInChildren<TMP_Text>(true);
            if (txt != null)
                txt.text = $"{d.classId}  {(d.score * 100f):0}%";

            box.gameObject.SetActive(true);
        }
        Debug.Log($"Overlay shown: {shown} / dets: {dets.Count}");
    }
}

---
name: tribe-v2
description: Analyze video/image brain responses using Meta's TRIBE v2 model. Use when predicting attention, emotion, or memory encoding for ads, content, or media. Triggers on "brain response", "neuromarketing", "attention analysis", "TRIBE", "predict brain", "ad engagement prediction".
---

# TRIBE v2 Brain Analysis

Predict human brain responses to video/image stimuli using Meta's TRIBE v2 foundation model.

## Quick Start

```bash
# Analyze a video
python scripts/tribe_analyze.py analyze https://example.com/video.mp4

# Analyze an image (auto-converts to 5s video)
python scripts/tribe_analyze.py analyze https://example.com/image.jpg

# Analyze local file
python scripts/tribe_analyze.py analyze /path/to/video.mp4
```

## Configuration

Set environment variables in `.env`:

```bash
# Option 1: RunPod Serverless (recommended)
TRIBE_RUNPOD_API_KEY=your_runpod_key
TRIBE_RUNPOD_ENDPOINT_ID=your_endpoint_id

# Option 2: HuggingFace Inference Endpoints (paid)
TRIBE_HF_ENDPOINT_URL=https://your-endpoint.huggingface.cloud
HF_TOKEN=hf_your_token

# Option 3: Local (requires GPU with 24GB+ VRAM)
TRIBE_LOCAL=true
HF_TOKEN=hf_your_token
```

## Output Format

```json
{
  "attention_curve": [0.45, 0.52, ...],
  "emotion_curve": [0.32, 0.41, ...],
  "memory_curve": [0.28, 0.35, ...],
  "overall_attention": 0.72,
  "overall_emotion": 0.68,
  "overall_memory": 0.75,
  "peak_moments": [
    {"time": 5, "type": "attention_spike", "score": 0.91}
  ],
  "duration_seconds": 30
}
```

## Brain Metrics Explained

| Metric | Brain Regions | Interpretation |
|--------|---------------|----------------|
| **Attention** | Visual cortex + Prefrontal | How much the content captures focus |
| **Emotion** | Limbic system | Emotional engagement level |
| **Memory** | Limbic + Prefrontal | Likelihood of being remembered |

## Peak Moment Types

- `attention_spike` - Sudden increase in attention (good hooks)
- `attention_drop` - Attention decrease (boring/confusing moments)
- `emotional_peak` - Strong emotional response
- `memory_encoding` - High memorability moment

## Use Cases

1. **Ad Testing** - Compare brain responses across ad variations
2. **Content Optimization** - Find attention drops to edit
3. **Thumbnail Selection** - Test which images grab attention
4. **Video Editing** - Identify best moments for clips

## Python API

```python
from tribe_analyze import TribeAnalyzer

analyzer = TribeAnalyzer()

# Analyze video
result = analyzer.analyze("https://example.com/video.mp4")

# Analyze with options
result = analyzer.analyze(
    "video.mp4",
    output_format="detailed",  # or "simple"
    save_curves=True           # save raw curves to file
)

# Get specific metrics
print(f"Attention: {result['overall_attention']:.0%}")
print(f"Peak moment: {result['peak_moments'][0]}")
```

## Limitations

- **License**: TRIBE v2 is CC-BY-NC-4.0 (non-commercial only)
- **Max duration**: 60 seconds recommended
- **GPU required**: 24GB+ VRAM for local inference
- **Image analysis**: Converts to 5-second video internally

## Cost (RunPod)

- ~$0.02-0.03 per analysis (60-90s GPU time)
- RTX 4090: $0.00031/second

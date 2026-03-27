# TRIBE v2 Skill for Clawdbot

A [Clawdbot](https://github.com/clawdbot/clawdbot) skill for analyzing video/image brain responses using Meta's TRIBE v2 foundation model.

## What is TRIBE v2?

[TRIBE v2](https://huggingface.co/facebook/tribev2) is Meta's brain encoding model that predicts fMRI brain responses to video, audio, and text stimuli. It enables "in-silico neuroscience" - predicting how humans will neurologically respond to content without actual brain scans.

## Features

- 🧠 **Attention Analysis** - Predict viewer attention over time
- ❤️ **Emotion Detection** - Identify emotional engagement peaks
- 💾 **Memory Encoding** - Predict which moments will be remembered
- 🖼️ **Image Support** - Automatically converts images to video for analysis
- 📊 **Multiple Backends** - RunPod, HuggingFace, or local inference

## Installation

### Via ClawdHub (recommended)
```bash
clawdhub install tribe-v2
```

### Manual
Copy the `tribe-v2` folder to your Clawdbot skills directory.

## Configuration

Add to your `.env`:

```bash
# Option 1: RunPod Serverless (recommended, ~$0.03/analysis)
TRIBE_RUNPOD_API_KEY=your_key
TRIBE_RUNPOD_ENDPOINT_ID=your_endpoint

# Option 2: HuggingFace Inference Endpoints
TRIBE_HF_ENDPOINT_URL=https://your-endpoint.huggingface.cloud
HF_TOKEN=hf_your_token

# Option 3: Local (requires 24GB+ GPU)
TRIBE_LOCAL=true
HF_TOKEN=hf_your_token
```

## Usage

### In Clawdbot
Just ask naturally:
- "Analyze this ad for brain response: https://example.com/ad.mp4"
- "What's the attention curve for this video?"
- "Predict engagement for this thumbnail"

### CLI
```bash
python scripts/tribe_analyze.py analyze https://example.com/video.mp4
python scripts/tribe_analyze.py analyze image.jpg --format detailed
```

### Python API
```python
from tribe_analyze import TribeAnalyzer

analyzer = TribeAnalyzer()
result = analyzer.analyze("video.mp4")

print(f"Attention: {result['overall_attention']:.0%}")
print(f"Emotion: {result['overall_emotion']:.0%}")
print(f"Memory: {result['overall_memory']:.0%}")
```

## Output Example

```json
{
  "attention_curve": [0.45, 0.52, 0.68, 0.82, ...],
  "emotion_curve": [0.32, 0.41, 0.55, 0.72, ...],
  "memory_curve": [0.28, 0.35, 0.42, 0.51, ...],
  "overall_attention": 0.72,
  "overall_emotion": 0.68,
  "overall_memory": 0.75,
  "peak_moments": [
    {"time": 5, "type": "attention_spike", "score": 0.91},
    {"time": 12, "type": "emotional_peak", "score": 0.85}
  ],
  "duration_seconds": 30
}
```

## Use Cases

- **Ad Testing** - Compare brain responses across ad variations
- **Content Optimization** - Find attention drops to edit
- **Thumbnail Selection** - Test which images grab attention
- **Video Editing** - Identify best moments for clips

## License

- This skill: MIT
- TRIBE v2 model: CC-BY-NC-4.0 (non-commercial use only)

## Links

- [TRIBE v2 on HuggingFace](https://huggingface.co/facebook/tribev2)
- [TRIBE v2 Paper](https://ai.meta.com/research/publications/a-foundation-model-of-vision-audition-and-language-for-in-silico-neuroscience/)
- [Clawdbot](https://github.com/clawdbot/clawdbot)

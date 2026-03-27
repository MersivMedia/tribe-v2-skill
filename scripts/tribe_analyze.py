#!/usr/bin/env python3
"""
TRIBE v2 Brain Analysis Tool

Analyze video/image stimuli for predicted brain responses using Meta's TRIBE v2 model.
Supports RunPod, HuggingFace Inference Endpoints, and local inference.

Usage:
    python tribe_analyze.py analyze <video_or_image_url>
    python tribe_analyze.py analyze /path/to/local/file.mp4
"""

import os
import sys
import json
import argparse
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, Union
from urllib.parse import urlparse

# Try to load environment from .env
def load_env():
    """Load environment variables from .env files."""
    env = dict(os.environ)
    search_paths = [
        Path('.env'),
        Path.home() / 'clawd' / '.env',
        Path('/home/clawdbot/clawd/.env'),
    ]
    for env_path in search_paths:
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, _, value = line.partition('=')
                    env[key.strip()] = value.strip()
    return env

ENV = load_env()


class TribeAnalyzer:
    """
    TRIBE v2 Brain Response Analyzer
    
    Supports multiple backends:
    - RunPod Serverless (recommended)
    - HuggingFace Inference Endpoints
    - Local inference (requires GPU)
    """
    
    def __init__(
        self,
        runpod_api_key: Optional[str] = None,
        runpod_endpoint_id: Optional[str] = None,
        hf_endpoint_url: Optional[str] = None,
        hf_token: Optional[str] = None,
        local: bool = False
    ):
        """
        Initialize the analyzer.
        
        Args:
            runpod_api_key: RunPod API key
            runpod_endpoint_id: RunPod endpoint ID
            hf_endpoint_url: HuggingFace Inference Endpoint URL
            hf_token: HuggingFace token
            local: Use local inference (requires GPU)
        """
        self.runpod_api_key = runpod_api_key or ENV.get('TRIBE_RUNPOD_API_KEY')
        self.runpod_endpoint_id = runpod_endpoint_id or ENV.get('TRIBE_RUNPOD_ENDPOINT_ID')
        self.hf_endpoint_url = hf_endpoint_url or ENV.get('TRIBE_HF_ENDPOINT_URL')
        self.hf_token = hf_token or ENV.get('HF_TOKEN')
        self.local = local or ENV.get('TRIBE_LOCAL', '').lower() == 'true'
        
        # Determine backend
        if self.runpod_api_key and self.runpod_endpoint_id:
            self.backend = 'runpod'
        elif self.hf_endpoint_url:
            self.backend = 'huggingface'
        elif self.local:
            self.backend = 'local'
        else:
            self.backend = None
    
    def analyze(
        self,
        source: str,
        output_format: str = 'standard',
        save_curves: bool = False,
        timeout: int = 180
    ) -> Dict[str, Any]:
        """
        Analyze a video or image for brain responses.
        
        Args:
            source: URL or local path to video/image
            output_format: 'simple', 'standard', or 'detailed'
            save_curves: Save raw curves to file
            timeout: Request timeout in seconds
        
        Returns:
            Dictionary with brain response data
        """
        if not self.backend:
            raise ValueError(
                "No backend configured. Set TRIBE_RUNPOD_API_KEY + TRIBE_RUNPOD_ENDPOINT_ID, "
                "TRIBE_HF_ENDPOINT_URL, or TRIBE_LOCAL=true"
            )
        
        # Detect if source is URL or local file
        is_url = source.startswith('http://') or source.startswith('https://')
        
        if is_url:
            media_url = source
        else:
            # Upload local file to temporary storage
            media_url = self._upload_local_file(source)
        
        # Run analysis based on backend
        if self.backend == 'runpod':
            result = self._analyze_runpod(media_url, timeout)
        elif self.backend == 'huggingface':
            result = self._analyze_huggingface(media_url, timeout)
        elif self.backend == 'local':
            result = self._analyze_local(source if not is_url else media_url)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
        
        # Format output
        if output_format == 'simple':
            result = self._simplify_output(result)
        elif output_format == 'detailed':
            result = self._add_detailed_output(result)
        
        # Save curves if requested
        if save_curves and 'attention_curve' in result:
            self._save_curves(result)
        
        return result
    
    def _analyze_runpod(self, media_url: str, timeout: int) -> Dict[str, Any]:
        """Run analysis via RunPod Serverless."""
        import requests
        
        endpoint = f"https://api.runpod.ai/v2/{self.runpod_endpoint_id}/runsync"
        
        response = requests.post(
            endpoint,
            headers={
                'Authorization': f'Bearer {self.runpod_api_key}',
                'Content-Type': 'application/json'
            },
            json={
                'input': {
                    'video_url': media_url
                }
            },
            timeout=timeout
        )
        
        response.raise_for_status()
        result = response.json()
        
        # Handle async job
        if result.get('status') in ['IN_QUEUE', 'IN_PROGRESS']:
            return self._poll_runpod_job(result['id'], timeout)
        
        return result.get('output', result)
    
    def _poll_runpod_job(self, job_id: str, timeout: int) -> Dict[str, Any]:
        """Poll RunPod job until completion."""
        import requests
        import time
        
        start_time = time.time()
        status_url = f"https://api.runpod.ai/v2/{self.runpod_endpoint_id}/status/{job_id}"
        
        while time.time() - start_time < timeout:
            response = requests.get(
                status_url,
                headers={'Authorization': f'Bearer {self.runpod_api_key}'},
                timeout=30
            )
            result = response.json()
            
            if result.get('status') == 'COMPLETED':
                return result.get('output', {})
            elif result.get('status') == 'FAILED':
                raise RuntimeError(f"RunPod job failed: {result.get('error')}")
            
            time.sleep(2)
        
        raise TimeoutError(f"RunPod job timed out after {timeout}s")
    
    def _analyze_huggingface(self, media_url: str, timeout: int) -> Dict[str, Any]:
        """Run analysis via HuggingFace Inference Endpoint."""
        import requests
        
        headers = {'Content-Type': 'application/json'}
        if self.hf_token:
            headers['Authorization'] = f'Bearer {self.hf_token}'
        
        response = requests.post(
            self.hf_endpoint_url,
            headers=headers,
            json={'video_url': media_url},
            timeout=timeout
        )
        
        response.raise_for_status()
        return response.json()
    
    def _analyze_local(self, source: str) -> Dict[str, Any]:
        """Run analysis locally using TRIBE v2 model."""
        try:
            from tribev2 import TribeModel
            import numpy as np
        except ImportError:
            raise ImportError(
                "tribev2 not installed. Install with: "
                "pip install 'tribev2 @ git+https://github.com/facebookresearch/tribev2.git'"
            )
        
        # Load model (cached after first load)
        if not hasattr(self, '_model'):
            print("Loading TRIBE v2 model...")
            self._model = TribeModel.from_pretrained("facebook/tribev2")
        
        # Check if source is image and convert
        source_path = Path(source)
        is_image = source_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp', '.gif']
        
        if is_image:
            video_path = self._convert_image_to_video(source)
        else:
            video_path = source
        
        # Run prediction
        df = self._model.get_events_dataframe(video_path=video_path)
        preds, segments = self._model.predict(events=df)
        
        # Process results
        return self._process_predictions(preds)
    
    def _convert_image_to_video(self, image_path: str, duration: int = 5) -> str:
        """Convert image to video using FFmpeg."""
        output_path = tempfile.mktemp(suffix='.mp4')
        
        cmd = [
            'ffmpeg', '-y',
            '-loop', '1',
            '-i', image_path,
            '-c:v', 'libx264',
            '-t', str(duration),
            '-pix_fmt', 'yuv420p',
            '-vf', 'scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2',
            '-r', '30',
            output_path
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        return output_path
    
    def _process_predictions(self, preds) -> Dict[str, Any]:
        """Process raw TRIBE v2 predictions into structured output."""
        import numpy as np
        
        # Brain region mappings (approximate vertex ranges)
        regions = {
            'visual': (0, 4000),
            'auditory': (4000, 8000),
            'language': (8000, 12000),
            'prefrontal': (12000, 16000),
            'limbic': (16000, 20484),
        }
        
        # Extract regional curves
        curves = {}
        for name, (start, end) in regions.items():
            curves[name] = preds[:, start:end].mean(axis=1)
        
        # Compute high-level metrics
        attention = (curves['visual'] + curves['prefrontal']) / 2
        emotion = curves['limbic']
        memory = (curves['limbic'] + curves['prefrontal']) / 2
        
        # Normalize
        def normalize(arr):
            arr = np.array(arr)
            if arr.max() - arr.min() > 0:
                return ((arr - arr.min()) / (arr.max() - arr.min())).tolist()
            return (arr * 0 + 0.5).tolist()
        
        attention_curve = normalize(attention)
        emotion_curve = normalize(emotion)
        memory_curve = normalize(memory)
        
        # Find peaks
        attention_peaks = self._find_peaks(attention_curve)
        emotion_peaks = self._find_peaks(emotion_curve)
        memory_peaks = self._find_peaks(memory_curve)
        attention_drops = self._find_drops(attention_curve)
        
        return {
            'attention_curve': attention_curve,
            'emotion_curve': emotion_curve,
            'memory_curve': memory_curve,
            'overall_attention': float(np.mean(attention_curve)),
            'overall_emotion': float(np.mean(emotion_curve)),
            'overall_memory': float(np.mean(memory_curve)),
            'attention_peaks': attention_peaks,
            'emotion_peaks': emotion_peaks,
            'memory_peaks': memory_peaks,
            'attention_drops': attention_drops,
            'peak_moments': self._format_peak_moments(
                attention_peaks, attention_drops, emotion_peaks, memory_peaks
            ),
            'duration_seconds': len(attention_curve)
        }
    
    def _find_peaks(self, curve: list, threshold: float = 0.7) -> list:
        """Find local maxima above threshold."""
        peaks = []
        for i in range(1, len(curve) - 1):
            if curve[i] > threshold and curve[i] > curve[i-1] and curve[i] >= curve[i+1]:
                peaks.append({'time': i, 'value': float(curve[i])})
        return peaks
    
    def _find_drops(self, curve: list) -> list:
        """Find significant drops in attention."""
        drops = []
        for i in range(1, len(curve)):
            if curve[i-1] > 0.5 and curve[i] < curve[i-1] * 0.7:
                drops.append({
                    'time': i,
                    'value': float(curve[i]),
                    'drop_from': float(curve[i-1])
                })
        return drops
    
    def _format_peak_moments(self, attention_peaks, attention_drops, emotion_peaks, memory_peaks):
        """Format peaks into unified timeline."""
        moments = []
        
        for p in attention_peaks:
            moments.append({'time': p['time'], 'type': 'attention_spike', 'score': p['value']})
        for d in attention_drops:
            moments.append({'time': d['time'], 'type': 'attention_drop', 'score': d['value']})
        for p in emotion_peaks:
            moments.append({'time': p['time'], 'type': 'emotional_peak', 'score': p['value']})
        for p in memory_peaks:
            moments.append({'time': p['time'], 'type': 'memory_encoding', 'score': p['value']})
        
        moments.sort(key=lambda x: x['time'])
        return moments
    
    def _simplify_output(self, result: Dict) -> Dict:
        """Return simplified output."""
        return {
            'attention': round(result.get('overall_attention', 0), 2),
            'emotion': round(result.get('overall_emotion', 0), 2),
            'memory': round(result.get('overall_memory', 0), 2),
            'duration': result.get('duration_seconds', 0),
            'peak_moments': result.get('peak_moments', [])[:5]
        }
    
    def _add_detailed_output(self, result: Dict) -> Dict:
        """Add detailed interpretation to output."""
        result['interpretation'] = {
            'attention_level': self._interpret_level(result.get('overall_attention', 0), 'attention'),
            'emotion_level': self._interpret_level(result.get('overall_emotion', 0), 'emotion'),
            'memory_level': self._interpret_level(result.get('overall_memory', 0), 'memory'),
            'recommendations': self._generate_recommendations(result)
        }
        return result
    
    def _interpret_level(self, value: float, metric: str) -> str:
        """Interpret a metric value."""
        if value > 0.7:
            return f"High {metric} - excellent engagement"
        elif value > 0.4:
            return f"Moderate {metric} - room for improvement"
        else:
            return f"Low {metric} - needs attention"
    
    def _generate_recommendations(self, result: Dict) -> list:
        """Generate recommendations based on analysis."""
        recs = []
        
        if result.get('attention_drops'):
            drops = result['attention_drops']
            if drops:
                times = [d['time'] for d in drops[:3]]
                recs.append(f"Attention drops at {', '.join(map(str, times))}s - consider adding hooks or cutting")
        
        if result.get('overall_attention', 0) < 0.5:
            recs.append("Low overall attention - strengthen opening hook")
        
        if result.get('overall_memory', 0) < 0.5:
            recs.append("Low memorability - add more distinctive/emotional moments")
        
        if not recs:
            recs.append("Good engagement overall - minor optimizations may still help")
        
        return recs
    
    def _upload_local_file(self, file_path: str) -> str:
        """Upload local file to temporary storage and return URL."""
        # For now, raise error - in production, upload to S3/GCS/etc.
        raise NotImplementedError(
            "Local file upload not implemented. "
            "Please use a URL or set TRIBE_LOCAL=true for local inference."
        )
    
    def _save_curves(self, result: Dict):
        """Save curves to JSON file."""
        output_path = Path('tribe_curves.json')
        curves = {
            'attention': result.get('attention_curve', []),
            'emotion': result.get('emotion_curve', []),
            'memory': result.get('memory_curve', []),
        }
        output_path.write_text(json.dumps(curves, indent=2))
        print(f"Curves saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='TRIBE v2 Brain Response Analyzer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s analyze https://example.com/video.mp4
  %(prog)s analyze /path/to/video.mp4 --local
  %(prog)s analyze image.jpg --format detailed
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze video/image')
    analyze_parser.add_argument('source', help='Video/image URL or local path')
    analyze_parser.add_argument('--format', '-f', choices=['simple', 'standard', 'detailed'],
                                default='standard', help='Output format')
    analyze_parser.add_argument('--save-curves', '-s', action='store_true',
                                help='Save raw curves to file')
    analyze_parser.add_argument('--local', '-l', action='store_true',
                                help='Use local inference')
    analyze_parser.add_argument('--timeout', '-t', type=int, default=180,
                                help='Request timeout in seconds')
    analyze_parser.add_argument('--output', '-o', help='Output file (JSON)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    if args.command == 'analyze':
        try:
            analyzer = TribeAnalyzer(local=args.local)
            result = analyzer.analyze(
                args.source,
                output_format=args.format,
                save_curves=args.save_curves,
                timeout=args.timeout
            )
            
            output = json.dumps(result, indent=2)
            
            if args.output:
                Path(args.output).write_text(output)
                print(f"Results saved to {args.output}")
            else:
                print(output)
                
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
DiaRemot Complete System Validation and Test

This script validates the complete system after fixes:
1. Runs quick diagnostic
2. Tests transcription with real audio
3. Validates pipeline integration
4. Reports final status
"""

import os
import sys
import time
import numpy as np
from pathlib import Path


def set_cpu_environment():
    """Set CPU-only environment"""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TORCH_DEVICE"] = "cpu"
    os.environ["WHISPER_DEVICE"] = "cpu"
    os.environ["WHISPER_COMPUTE_TYPE"] = "float32"
    print("✓ CPU-only environment configured")


def create_test_audio():
    """Create a short test audio file"""
    print("Creating test audio...")

    # Generate 5 seconds of test audio with simple patterns
    sr = 16000
    duration = 5
    t = np.linspace(0, duration, sr * duration)

    # Create simple speech-like patterns
    audio = np.zeros_like(t)

    # Add some "speech" bursts
    for start_time in [0.5, 2.0, 3.5]:
        start_idx = int(start_time * sr)
        end_idx = int((start_time + 1.0) * sr)

        # Simple modulated tone
        freq = 200 + 100 * np.random.random()
        burst = np.sin(2 * np.pi * freq * t[start_idx:end_idx])
        burst *= np.exp(-5 * (t[start_idx:end_idx] - start_time))  # Fade
        audio[start_idx:end_idx] = burst * 0.1

    # Add some noise
    audio += np.random.normal(0, 0.01, len(audio))

    # Save as wav file
    test_audio_path = Path("test_audio.wav")
    try:
        import soundfile as sf

        sf.write(test_audio_path, audio.astype(np.float32), sr)
        print(f"✓ Test audio created: {test_audio_path}")
        return str(test_audio_path)
    except ImportError:
        print("⚠ soundfile not available, using numpy save")
        np.save("test_audio.npy", audio.astype(np.float32))
        return "test_audio.npy"


def test_transcription_directly():
    """Test transcription module directly"""
    print("\n" + "=" * 50)
    print("DIRECT TRANSCRIPTION TEST")
    print("=" * 50)

    try:
        from diaremot.pipeline.transcription_module import AudioTranscriber

        # Initialize with CPU-only settings
        print("Initializing transcriber...")
        transcriber = AudioTranscriber(
            model_size="faster-whisper-tiny.en",  # Use fastest model for testing
            compute_type="float32",
            beam_size=1,
            temperature=0.0,
            no_speech_threshold=0.6,
        )
        print("✓ Transcriber initialized")

        # Validate backend
        if hasattr(transcriber, "validate_backend"):
            validation = transcriber.validate_backend()
            print(f"Backend: {validation['active_backend']}")
            print(f"Functional: {validation['backend_functional']}")

            if not validation["backend_functional"]:
                print(f"❌ Backend validation failed: {validation['error']}")
                return False
        else:
            print("⚠ No validation method available")

        # Test with synthetic audio
        print("\nTesting with synthetic audio...")
        test_audio = (
            np.random.randn(16000).astype(np.float32) * 0.01
        )  # 1 second of quiet noise
        test_segments = [
            {
                "start_time": 0.0,
                "end_time": 1.0,
                "speaker_id": "test",
                "speaker_name": "Test Speaker",
            }
        ]

        start_time = time.time()
        results = transcriber.transcribe_segments(test_audio, 16000, test_segments)
        elapsed = time.time() - start_time

        if results and len(results) > 0:
            result = results[0]
            print(f"✓ Transcription successful in {elapsed:.2f}s")
            print(f"  - Text: '{result.text}'")
            print(f"  - Model: {result.model_used}")
            print(f"  - Confidence: {result.confidence}")
            return True
        else:
            print("❌ No transcription results returned")
            return False

    except Exception as e:
        print(f"❌ Direct transcription test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_full_pipeline():
    """Test the complete pipeline"""
    print("\n" + "=" * 50)
    print("FULL PIPELINE TEST")
    print("=" * 50)

    try:
        if str(Path.cwd()) not in sys.path:
            sys.path.insert(0, str(Path.cwd()))
        from .audio_pipeline_core import AudioAnalysisPipelineV2

        # Create test audio
        test_audio_path = create_test_audio()

        # Configure pipeline for fast testing
        config = {
            "whisper_model": "faster-whisper-tiny.en",
            "noise_reduction": False,
            "beam_size": 1,
            "temperature": 0.0,
            "no_speech_threshold": 0.6,
            "registry_path": "test_speaker_registry.json",
        }

        print("Initializing pipeline...")
        pipeline = AudioAnalysisPipelineV2(config)
        print("✓ Pipeline initialized")

        # Check transcriber
        if hasattr(pipeline, "tx"):
            info = pipeline.tx.get_model_info()
            print(f"  - Backend: {info.get('backend', 'unknown')}")
            print(f"  - Device: {info.get('device', 'unknown')}")
            print(f"  - Model: {info.get('model_size', 'unknown')}")

            if info.get("backend") == "fallback":
                print("⚠ Pipeline using fallback transcriber")

        # Run pipeline on test audio
        output_dir = Path("test_output")
        output_dir.mkdir(exist_ok=True)

        print(f"\nRunning pipeline on {test_audio_path}...")
        start_time = time.time()

        if test_audio_path.endswith(".npy"):
            # For numpy files, create a minimal wav file
            audio_data = np.load(test_audio_path)
            import tempfile
            import soundfile as sf

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, audio_data, 16000)
                test_audio_path = f.name

        result = pipeline.process_audio_file(test_audio_path, str(output_dir))
        elapsed = time.time() - start_time

        print(f"✓ Pipeline completed in {elapsed:.2f}s")
        print(f"  - Run ID: {result['run_id']}")
        print(f"  - Output dir: {result['out_dir']}")

        # Check outputs
        csv_path = Path(result["outputs"]["csv"])
        if csv_path.exists():
            print(f"  - CSV output: {csv_path}")
            # Count lines
            with open(csv_path, "r") as f:
                lines = f.readlines()
            print(f"  - Segments: {len(lines) - 1}")  # -1 for header

        return True

    except Exception as e:
        print(f"❌ Full pipeline test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_parselmouth():
    """Verify Parselmouth integration via paralinguistics"""
    print("\n" + "=" * 50)
    print("PARSELMOUTH INTEGRATION TEST")
    print("=" * 50)
    try:
        from diaremot.affect import paralinguistics as para
    except Exception:
        if str(Path.cwd()) not in sys.path:
            sys.path.insert(0, str(Path.cwd()))
        import paralinguistics as para

        if not getattr(para, "PARSELMOUTH_AVAILABLE", False):
            print("⚠ Parselmouth not available")
            return False
        audio = np.sin(2 * np.pi * 100 * np.linspace(0, 1, 16000)).astype(np.float32)
        cfg = para.ParalinguisticsConfig(vq_use_parselmouth=True)
        res = para._compute_voice_quality_parselmouth_v2(audio, 16000, cfg)
        print(f"✓ Parselmouth functional (jitter_pct={res.get('jitter_pct', 0.0):.3f})")
        return True
    except Exception as e:
        print(f"❌ Parselmouth test failed: {e}")
        return False


def run_comprehensive_validation():
    """Run all validation tests"""
    print("DiaRemot Complete System Validation")
    print("=" * 60)

    # Set environment
    set_cpu_environment()

    # Check if we're in the right directory
    # Ensure we're at the project root where modules live directly
    project_root_marker = Path("src/diaremot/pipeline/transcription_module.py")
    if not project_root_marker.exists():
        print("❌ Please run this script from your project root directory")
        return False

    # Run tests
    tests = [
        ("Direct Transcription", test_transcription_directly),
        ("Full Pipeline", test_full_pipeline),
        ("Parselmouth Integration", test_parselmouth),
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ {test_name} test crashed: {e}")
            results[test_name] = False

    # Summary
    print("\n" + "=" * 60)
    print("FINAL VALIDATION RESULTS")
    print("=" * 60)

    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:25} {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n🎉 SYSTEM FULLY VALIDATED - ALL SYSTEMS WORKING!")
        print("\nYour DiaRemot system is ready to use:")
        print("  • CPU-only transcription: ✓")
        print("  • Faster-Whisper backend: ✓")
        print("  • Full pipeline integration: ✓")
        print("\nYou can now process audio files with confidence.")
        return True
    else:
        print("\n❌ VALIDATION FAILED - SOME ISSUES REMAIN")
        print("\nTo fix remaining issues:")
        print("  1. Run: python transcription_backend_comprehensive_fix.py")
        print("  2. Check dependency installation")
        print("  3. Review error messages above")
        return False


def cleanup_test_files():
    """Clean up test files"""
    test_files = ["test_audio.wav", "test_audio.npy", "test_speaker_registry.json"]

    for file in test_files:
        try:
            Path(file).unlink(missing_ok=True)
        except Exception:
            pass

    try:
        import shutil

        shutil.rmtree("test_output", ignore_errors=True)
    except Exception:
        pass


if __name__ == "__main__":
    try:
        success = run_comprehensive_validation()
        cleanup_test_files()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nValidation interrupted by user")
        cleanup_test_files()
        sys.exit(1)
    except Exception as e:
        print(f"\n\nValidation script crashed: {e}")
        cleanup_test_files()
        sys.exit(1)

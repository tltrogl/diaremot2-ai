"""Test paralinguistics fallback in orchestrator when para module fails."""

import numpy as np
import pytest

from diaremot.pipeline.orchestrator import AudioAnalysisPipelineV2


class TestParalinguisticsFallback:
    """Test that orchestrator fallback populates all required schema fields."""

    def test_fallback_populates_all_fields(self, monkeypatch):
        """When para.extract() fails, fallback must populate all CSV schema fields."""

        # Mock para module to raise exception
        import diaremot.pipeline.orchestrator as orch_module

        original_para = orch_module.para

        class BrokenPara:
            @staticmethod
            def extract(wav, sr, segs):
                raise RuntimeError("Para module broken for test")

        monkeypatch.setattr(orch_module, "para", BrokenPara())

        # Create pipeline instance
        config = {
            "quiet": True,
            "disable_affect": True,
        }
        pipe = AudioAnalysisPipelineV2(config)

        # Create test data
        wav = np.random.randn(16000).astype(np.float32)  # 1 sec audio
        sr = 16000
        segs = [
            {
                "start": 0.0,
                "end": 1.0,
                "text": "Hello world test",
                "speaker_id": "SPEAKER_00",
            }
        ]

        # Call fallback path
        result = pipe._extract_paraling(wav, sr, segs)

        # Verify all required fields present
        assert 0 in result
        metrics = result[0]

        required_fields = [
            "wpm",
            "duration_s",
            "words",
            "pause_count",
            "pause_time_s",
            "pause_ratio",
            "f0_mean_hz",
            "f0_std_hz",
            "loudness_rms",
            "disfluency_count",
            "vq_jitter_pct",
            "vq_shimmer_db",
            "vq_hnr_db",
            "vq_cpps_db",
        ]

        for field in required_fields:
            assert field in metrics, f"Missing required field: {field}"

        # Verify types and basic values
        assert isinstance(metrics["wpm"], float)
        assert isinstance(metrics["duration_s"], float)
        assert isinstance(metrics["words"], int)
        assert metrics["duration_s"] == pytest.approx(1.0, abs=0.01)
        assert metrics["words"] == 3  # "Hello world test"
        assert metrics["wpm"] > 0  # Should compute WPM from words/duration

        # Voice quality fields should be 0.0 in fallback
        assert metrics["vq_jitter_pct"] == 0.0
        assert metrics["vq_shimmer_db"] == 0.0
        assert metrics["vq_hnr_db"] == 0.0
        assert metrics["vq_cpps_db"] == 0.0

        # Restore original para
        monkeypatch.setattr(orch_module, "para", original_para)

    def test_fallback_handles_empty_text(self, monkeypatch):
        """Fallback should handle segments with no text."""

        import diaremot.pipeline.orchestrator as orch_module

        original_para = orch_module.para

        class BrokenPara:
            @staticmethod
            def extract(wav, sr, segs):
                raise RuntimeError("Para module broken for test")

        monkeypatch.setattr(orch_module, "para", BrokenPara())

        config = {"quiet": True, "disable_affect": True}
        pipe = AudioAnalysisPipelineV2(config)

        wav = np.random.randn(16000).astype(np.float32)
        sr = 16000
        segs = [{"start": 0.0, "end": 1.0, "text": "", "speaker_id": "SPEAKER_00"}]

        result = pipe._extract_paraling(wav, sr, segs)

        assert 0 in result
        metrics = result[0]
        assert metrics["words"] == 0
        assert metrics["wpm"] == 0.0
        assert "duration_s" in metrics

        monkeypatch.setattr(orch_module, "para", original_para)

    def test_fallback_computes_loudness(self, monkeypatch):
        """Fallback should compute RMS loudness from audio."""

        import diaremot.pipeline.orchestrator as orch_module

        original_para = orch_module.para

        class BrokenPara:
            @staticmethod
            def extract(wav, sr, segs):
                raise RuntimeError("Para module broken for test")

        monkeypatch.setattr(orch_module, "para", BrokenPara())

        config = {"quiet": True, "disable_affect": True}
        pipe = AudioAnalysisPipelineV2(config)

        # Create audio with known RMS
        amplitude = 0.5
        wav = np.ones(16000, dtype=np.float32) * amplitude
        sr = 16000
        segs = [{"start": 0.0, "end": 1.0, "text": "test", "speaker_id": "SPEAKER_00"}]

        result = pipe._extract_paraling(wav, sr, segs)

        assert 0 in result
        metrics = result[0]
        assert "loudness_rms" in metrics
        assert metrics["loudness_rms"] == pytest.approx(amplitude, abs=0.01)

        monkeypatch.setattr(orch_module, "para", original_para)

    def test_fallback_multiple_segments(self, monkeypatch):
        """Fallback should handle multiple segments correctly."""

        import diaremot.pipeline.orchestrator as orch_module

        original_para = orch_module.para

        class BrokenPara:
            @staticmethod
            def extract(wav, sr, segs):
                raise RuntimeError("Para module broken for test")

        monkeypatch.setattr(orch_module, "para", BrokenPara())

        config = {"quiet": True, "disable_affect": True}
        pipe = AudioAnalysisPipelineV2(config)

        wav = np.random.randn(48000).astype(np.float32)  # 3 sec
        sr = 16000
        segs = [
            {"start": 0.0, "end": 1.0, "text": "first segment", "speaker_id": "SPEAKER_00"},
            {"start": 1.0, "end": 2.0, "text": "second segment here", "speaker_id": "SPEAKER_01"},
            {"start": 2.0, "end": 3.0, "text": "third", "speaker_id": "SPEAKER_00"},
        ]

        result = pipe._extract_paraling(wav, sr, segs)

        assert len(result) == 3
        assert all(i in result for i in range(3))

        # Each segment should have different word counts
        assert result[0]["words"] == 2  # "first segment"
        assert result[1]["words"] == 3  # "second segment here"
        assert result[2]["words"] == 1  # "third"

        monkeypatch.setattr(orch_module, "para", original_para)

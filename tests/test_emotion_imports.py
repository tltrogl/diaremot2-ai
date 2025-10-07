"""Regression tests for emotion analysis import compatibility."""

def test_affect_package_exposes_emotion_analysis():
    from diaremot.affect import emotion_analysis

    assert hasattr(emotion_analysis, "EmotionAnalyzer")


def test_root_package_legacy_emotion_analysis_alias():
    from diaremot import emotion_analysis

    assert hasattr(emotion_analysis, "EmotionAnalyzer")

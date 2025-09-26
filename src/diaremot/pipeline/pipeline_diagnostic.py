#!/usr/bin/env python
"""Comprehensive pipeline diagnostic and fix utility.

Diagnostic helpers for validating DiaRemot pipeline prerequisites.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple


def _module_available(module_name: str) -> bool:
    """Return ``True`` if ``module_name`` can be imported."""

    return importlib.util.find_spec(module_name) is not None

class PipelineDiagnostic:
    def __init__(self):
        self.issues = []
        self.fixes_applied = []
        self.warnings = []
        
    def check_cache_issues(self) -> Tuple[bool, str]:
        """Check for cache-related issues"""
        cache_dir = Path(".cache")
        if not cache_dir.exists():
            return True, "No cache directory found (clean state)"
        
        cache_files = list(cache_dir.rglob("*.json"))
        if cache_files:
            self.warnings.append(f"Found {len(cache_files)} cache files that may prevent fresh processing")
            return False, f"Cache files present: {len(cache_files)} files"
        return True, "Cache directory empty"
    
    def clear_cache(self) -> bool:
        """Clear all cache files"""
        try:
            cache_dir = Path(".cache")
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
                cache_dir.mkdir(parents=True, exist_ok=True)
                self.fixes_applied.append("Cache cleared successfully")
                return True
            return True
        except Exception as e:
            self.issues.append(f"Failed to clear cache: {e}")
            return False
    
    def check_audio_dependencies(self) -> Tuple[bool, str]:
        """Check audio processing dependencies"""
        issues = []
        
        # Check soundfile
        try:
            import soundfile
            # Try to check if libsndfile is actually working
            try:
                import numpy as np
                test_audio = np.zeros(100)
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as f:
                    soundfile.write(f.name, test_audio, 16000)
                    soundfile.read(f.name)
            except Exception as e:
                issues.append(f"soundfile installed but not working: {e}")
        except ImportError:
            issues.append("soundfile not installed")
        
        # Check librosa
        if not _module_available("librosa"):
            issues.append("librosa not installed")

        # Check audioread (fallback)
        if not _module_available("audioread"):
            self.warnings.append("audioread not installed (fallback audio loader)")
        
        # Check ffmpeg
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                issues.append("ffmpeg not working properly")
        except FileNotFoundError:
            issues.append("ffmpeg not found in PATH")
        except Exception as e:
            issues.append(f"ffmpeg check failed: {e}")
        
        if issues:
            self.issues.extend(issues)
            return False, f"Audio dependencies issues: {', '.join(issues)}"
        return True, "Audio dependencies OK"
    
    def fix_audio_dependencies(self) -> bool:
        """Try to fix audio dependency issues"""
        fixed = False
        
        # Try to reinstall critical packages
        packages = [
            ("soundfile", "0.12.1"),
            ("librosa", "0.10.1"),
            ("audioread", None),
            ("ffmpeg-python", None)
        ]
        
        for package, version in packages:
            try:
                if version:
                    cmd = [sys.executable, "-m", "pip", "install", f"{package}=={version}", "--force-reinstall"]
                else:
                    cmd = [sys.executable, "-m", "pip", "install", package]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    self.fixes_applied.append(f"Reinstalled {package}")
                    fixed = True
            except Exception as e:
                self.warnings.append(f"Could not reinstall {package}: {e}")
        
        return fixed
    
    def check_model_dependencies(self) -> Tuple[bool, str]:
        """Check ML model dependencies"""
        issues = []
        
        # Check PyTorch
        if _module_available("torch"):
            torch = importlib.import_module("torch")
            if not getattr(torch, "__version__", ""):
                issues.append("PyTorch version unknown")
        else:
            issues.append("PyTorch not installed")

        # Check transformers
        if not _module_available("transformers"):
            issues.append("transformers not installed")

        # Check ONNX runtime
        if not _module_available("onnxruntime"):
            self.warnings.append("onnxruntime not installed (optional for CPU optimization)")

        # Check faster-whisper
        if not _module_available("faster_whisper"):
            issues.append("faster-whisper not installed")

        # Check ctranslate2
        if not _module_available("ctranslate2"):
            issues.append("ctranslate2 not installed")
        
        if issues:
            self.issues.extend(issues)
            return False, f"Model dependencies issues: {', '.join(issues)}"
        return True, "Model dependencies OK"
    
    def check_pipeline_config(self) -> Tuple[bool, str]:
        """Check pipeline configuration"""
        config_file = Path("pipeline_config.json")
        
        if not config_file.exists():
            self.warnings.append("No pipeline_config.json found (using defaults)")
            return True, "Using default configuration"
        
        try:
            with open(config_file) as f:
                config = json.load(f)
            
            # Check for common misconfigurations
            if config.get("device") == "cuda" and not self._cuda_available():
                self.warnings.append("Config specifies CUDA but GPU not available")
            
            return True, "Configuration loaded successfully"
        except Exception as e:
            self.issues.append(f"Config file error: {e}")
            return False, f"Configuration error: {e}"
    
    def _cuda_available(self) -> bool:
        """Check if CUDA is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False
    
    def run_diagnostics(self) -> Dict:
        """Run all diagnostics"""
        print("="*60)
        print("PIPELINE DIAGNOSTIC TOOL")
        print("="*60)
        print("\nRunning diagnostics...")
        print("-"*40)
        
        results = {}
        
        # Check cache
        cache_ok, cache_msg = self.check_cache_issues()
        results['cache'] = {'ok': cache_ok, 'message': cache_msg}
        print(f"{'✓' if cache_ok else '✗'} Cache: {cache_msg}")
        
        # Check audio deps
        audio_ok, audio_msg = self.check_audio_dependencies()
        results['audio'] = {'ok': audio_ok, 'message': audio_msg}
        print(f"{'✓' if audio_ok else '✗'} Audio: {audio_msg}")
        
        # Check model deps
        model_ok, model_msg = self.check_model_dependencies()
        results['models'] = {'ok': model_ok, 'message': model_msg}
        print(f"{'✓' if model_ok else '✗'} Models: {model_msg}")
        
        # Check config
        config_ok, config_msg = self.check_pipeline_config()
        results['config'] = {'ok': config_ok, 'message': config_msg}
        print(f"{'✓' if config_ok else '✗'} Config: {config_msg}")
        
        return results
    
    def apply_fixes(self) -> bool:
        """Apply automatic fixes"""
        print("\n" + "="*60)
        print("APPLYING FIXES")
        print("-"*40)
        
        any_fixed = False
        
        # Clear cache if needed
        if any("cache" in issue.lower() for issue in self.issues + self.warnings):
            print("Clearing cache...")
            if self.clear_cache():
                print("  ✓ Cache cleared")
                any_fixed = True
        
        # Fix audio dependencies if needed
        if any("audio" in issue.lower() or "soundfile" in issue.lower() 
               for issue in self.issues):
            print("Fixing audio dependencies...")
            if self.fix_audio_dependencies():
                print("  ✓ Audio dependencies updated")
                any_fixed = True
        
        return any_fixed
    
    def generate_report(self, results: Dict) -> None:
        """Generate diagnostic report"""
        print("\n" + "="*60)
        print("DIAGNOSTIC REPORT")
        print("-"*40)
        
        all_ok = all(r['ok'] for r in results.values())
        
        if self.issues:
            print("\n❌ CRITICAL ISSUES:")
            for issue in self.issues:
                print(f"  • {issue}")
        
        if self.warnings:
            print("\n⚠️  WARNINGS:")
            for warning in self.warnings:
                print(f"  • {warning}")
        
        if self.fixes_applied:
            print("\n✅ FIXES APPLIED:")
            for fix in self.fixes_applied:
                print(f"  • {fix}")
        
        print("\n" + "="*60)
        if all_ok and not self.issues:
            print("✓ PIPELINE READY - All checks passed!")
        else:
            print("⚠ ISSUES DETECTED - Review above and apply manual fixes if needed")
            print("\nSuggested manual fixes:")
            
            if any("ffmpeg" in issue.lower() for issue in self.issues):
                print("1. Install FFmpeg:")
                print("   • Download from: https://ffmpeg.org/download.html")
                print("   • Add to system PATH")
            
            if any("soundfile" in issue.lower() for issue in self.issues):
                print("2. Install libsndfile (Windows):")
                print("   • Via conda: conda install -c conda-forge libsndfile")
                print("   • Or download from: https://github.com/libsndfile/libsndfile/releases")
        
        # Save report to file
        report_path = Path("diagnostic_report.json")
        with open(report_path, "w") as f:
            json.dump({
                "results": results,
                "issues": self.issues,
                "warnings": self.warnings,
                "fixes_applied": self.fixes_applied
            }, f, indent=2)
        
        print(f"\nDetailed report saved to: {report_path}")

def main():
    diagnostic = PipelineDiagnostic()
    
    # Run diagnostics
    results = diagnostic.run_diagnostics()
    
    # Apply fixes if needed
    if diagnostic.issues:
        response = input("\nApply automatic fixes? (y/n): ").lower()
        if response == 'y':
            diagnostic.apply_fixes()
            
            # Re-run diagnostics
            print("\nRe-running diagnostics after fixes...")
            diagnostic = PipelineDiagnostic()
            results = diagnostic.run_diagnostics()
    
    # Generate report
    diagnostic.generate_report(results)
    
    # Offer to run test
    if not diagnostic.issues:
        response = input("\nRun a test audio processing? (y/n): ").lower()
        if response == 'y':
            print("\nRunning test...")
            try:
                subprocess.run([
                    sys.executable,
                    "-m",
                    "diaremot.pipeline.audio_pipeline_core",
                    "--verify_deps"
                ])
            except Exception as e:
                print(f"Test failed: {e}")

if __name__ == "__main__":
    main()

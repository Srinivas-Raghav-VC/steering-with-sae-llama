#!/usr/bin/env python3
"""
Quick CPU sanity check for the pipeline without requiring GPU or HF tokens.
Tests basic functionality, imports, and data structures.
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all critical imports work."""
    try:
        from he_steer_pipeline import (
            Config, 
            setup_environment,
            devanagari_ratio,
            detect_language_simple,
            JumpReLUSAE
        )
        print("✅ Core pipeline imports successful")
        return True
    except Exception as e:
        print(f"❌ Pipeline import failed: {e}")
        return False

def test_config():
    """Test that Config dataclass works."""
    try:
        from he_steer_pipeline import Config
        cfg = Config(
            layer_range=(19, 21),  # correct parameter name
            samples_per_language=100,  # small for test
            train_epochs=5,
            eval_prompts=10
        )
        print(f"✅ Config creation successful: layer_range={cfg.layer_range}")
        return True
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

def test_language_detection():
    """Test language detection functions."""
    try:
        from he_steer_pipeline import devanagari_ratio, detect_language_simple
        
        # Test devanagari detection
        hindi_text = "नमस्ते दुनिया"
        english_text = "Hello world"
        mixed_text = "Hello नमस्ते world"
        
        hindi_ratio = devanagari_ratio(hindi_text)
        english_ratio = devanagari_ratio(english_text)
        mixed_ratio = devanagari_ratio(mixed_text)
        
        print(f"✅ Devanagari ratios: Hindi={hindi_ratio:.2f}, English={english_ratio:.2f}, Mixed={mixed_ratio:.2f}")
        
        # Test language classification
        hindi_lang = detect_language_simple(hindi_text)
        english_lang = detect_language_simple(english_text)
        
        print(f"✅ Language detection: '{hindi_text}' -> {hindi_lang}, '{english_text}' -> {english_lang}")
        return True
    except Exception as e:
        print(f"❌ Language detection test failed: {e}")
        return False

def test_sae_class():
    """Test that SAE class can be instantiated."""
    try:
        from he_steer_pipeline import JumpReLUSAE
        import torch
        
        sae = JumpReLUSAE(input_dim=128, expansion_factor=4, l0_target=100)
        print(f"✅ SAE instantiation successful: input_dim={sae.input_dim}, hidden_dim={sae.hidden_dim}")
        
        # Test basic forward pass with dummy data
        dummy_input = torch.randn(2, 10, 128)  # batch=2, seq=10, input_dim=128
        with torch.no_grad():
            recon, l0_loss, l1_loss = sae(dummy_input)
        
        print(f"✅ SAE forward pass successful: output_shape={recon.shape}")
        return True
    except Exception as e:
        print(f"❌ SAE test failed: {e}")
        return False

def test_evaluation_imports():
    """Test evaluation and downstream tools."""
    try:
        import evaluation_improved
        from evaluation_improved import classify_language, EvaluationResult
        
        # Test language classification
        result = classify_language("Hello world")
        print(f"✅ Evaluation language classification: 'Hello world' -> {result}")
        
        import linear_probe_baseline
        print("✅ Linear probe baseline imports successfully")
        
        import rescore_results
        print("✅ Rescore results imports successfully")
        
        return True
    except Exception as e:
        print(f"❌ Evaluation imports failed: {e}")
        return False

def main():
    """Run all sanity checks."""
    print("🧪 Running CPU sanity checks...\n")
    
    tests = [
        ("Imports", test_imports),
        ("Config", test_config),
        ("Language Detection", test_language_detection),
        ("SAE Class", test_sae_class),
        ("Evaluation Tools", test_evaluation_imports),
    ]
    
    passed = 0
    total = len(tests)
    
    for name, test_func in tests:
        print(f"\n📋 Testing {name}:")
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {name} test failed")
        except Exception as e:
            print(f"❌ {name} test crashed: {e}")
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All sanity checks passed! Pipeline is ready for GPU runs.")
    else:
        print("⚠️  Some tests failed. Review errors above before proceeding.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
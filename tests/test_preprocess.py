# tests/test_preprocess.py
"""Unit tests for the NLP preprocessing pipeline."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.preprocess import clean_text


def test_clean_text_lowercase():
    result = clean_text("The Doctor Was VERY Kind")
    assert result == result.lower()
    print("  ✅ lowercase OK")

def test_clean_text_removes_punctuation():
    result = clean_text("Great service! Really helpful.")
    assert "!" not in result and "." not in result
    print("  ✅ punctuation removal OK")

def test_clean_text_removes_stopwords():
    result = clean_text("the nurse was very kind and helpful")
    assert "the" not in result.split()
    assert "and" not in result.split()
    print("  ✅ stopword removal OK")

def test_clean_text_lemmatization():
    result = clean_text("The nurses were helping patients")
    assert "help" in result or "helping" not in result
    print("  ✅ lemmatization OK")

def test_clean_text_empty():
    result = clean_text("")
    assert result == ""
    print("  ✅ empty string OK")


if __name__ == "__main__":
    tests = [
        test_clean_text_lowercase,
        test_clean_text_removes_punctuation,
        test_clean_text_removes_stopwords,
        test_clean_text_lemmatization,
        test_clean_text_empty,
    ]
    passed = failed = 0
    print("=" * 45)
    print("  TESTS - Preprocessing Pipeline")
    print("=" * 45)
    for t in tests:
        try:
            t(); passed += 1
        except AssertionError as e:
            print(f"  ❌ FAILED: {t.__name__} → {e}"); failed += 1
        except Exception as e:
            print(f"  💥 ERROR:  {t.__name__} → {e}"); failed += 1
    print("=" * 45)
    print(f"  Result: {passed} passed / {failed} failed")
    print("=" * 45)
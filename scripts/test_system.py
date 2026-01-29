"""
Test VinaCompare system
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import json
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:8000"


def test_health_check():
    """Test health check endpoint"""
    logger.info("Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        assert response.status_code == 200
        data = response.json()
        logger.info(f"Health check: {json.dumps(data, indent=2)}")
        return True
    except requests.exceptions.ConnectionError:
        logger.error("Connection failed. Is the server running?")
        return False
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return False


def test_root():
    """Test root endpoint"""
    logger.info("Testing root endpoint...")
    response = requests.get(f"{BASE_URL}/")
    assert response.status_code == 200
    data = response.json()
    logger.info(f"Root: {json.dumps(data, indent=2)}")
    return True


def test_models():
    """Test models endpoint"""
    logger.info("Testing models endpoint...")
    response = requests.get(f"{BASE_URL}/api/v1/models")
    assert response.status_code == 200
    data = response.json()
    logger.info(f"Available models: {json.dumps(data, indent=2)}")
    return True


def test_stats():
    """Test stats endpoint"""
    logger.info("Testing stats endpoint...")
    response = requests.get(f"{BASE_URL}/api/v1/stats")
    assert response.status_code == 200
    data = response.json()
    logger.info(f"Stats: {json.dumps(data, indent=2)}")
    return True


def test_search():
    """Test search endpoint"""
    logger.info("Testing search endpoint...")

    payload = {
        "query": "Python là gì?",
        "top_k": 3,
        "mode": "dense"
    }

    response = requests.post(
        f"{BASE_URL}/api/v1/search",
        json=payload
    )

    if response.status_code == 200:
        data = response.json()
        logger.info(f"Search results: {data['count']} documents found")
        for i, result in enumerate(data['results'][:3]):
            logger.info(f"  [{i+1}] {result['document_id']}: score={result['score']:.4f}")
            logger.info(f"      {result['text'][:100]}...")
        return True
    else:
        logger.warning(f"Search failed: {response.status_code} - {response.text}")
        return False


def test_query():
    """Test RAG query"""
    logger.info("Testing RAG query (this may take a while if model is loading)...")

    payload = {
        "question": "Python là gì và cách bắt đầu học Python?",
        "model": "Vistral-7B-Chat",
        "top_k": 3,
        "retrieval_mode": "dense"
    }

    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/query",
            json=payload,
            timeout=300  # 5 minute timeout for model loading
        )

        if response.status_code == 200:
            result = response.json()
            logger.info(f"Query ID: {result['query_id']}")
            logger.info(f"Question: {result['question']}")
            logger.info(f"Answer: {result['answer'][:500]}...")
            logger.info(f"Confidence: {result['confidence']}")
            logger.info(f"Sources: {len(result['sources'])}")
            logger.info(f"Metrics:")
            logger.info(f"  - Retrieval time: {result['metrics']['retrieval_time_ms']}ms")
            logger.info(f"  - Generation time: {result['metrics']['generation_time_ms']}ms")
            logger.info(f"  - Total time: {result['metrics']['total_time_ms']}ms")
            return True
        elif response.status_code == 503:
            logger.warning("Model not loaded yet. RAG query skipped.")
            return None
        else:
            logger.error(f"Query failed: {response.status_code} - {response.text}")
            return False

    except requests.exceptions.Timeout:
        logger.warning("Query timed out. Model may still be loading.")
        return None
    except Exception as e:
        logger.error(f"Query failed: {e}")
        return False


def main():
    """Run all tests"""
    logger.info("=" * 50)
    logger.info("Testing VinaCompare System")
    logger.info("=" * 50)

    results = {}

    # Test connection first
    if not test_health_check():
        logger.error("Server not responding. Please start the server first:")
        logger.error("  cd D:/vinacompare && python src/main.py")
        return

    # Run tests
    tests = [
        ("Root", test_root),
        ("Models", test_models),
        ("Stats", test_stats),
        ("Search", test_search),
        ("RAG Query", test_query),
    ]

    for name, test_func in tests:
        try:
            result = test_func()
            results[name] = result
        except Exception as e:
            logger.error(f"{name} test failed with exception: {e}")
            results[name] = False

    # Summary
    logger.info("")
    logger.info("=" * 50)
    logger.info("Test Summary")
    logger.info("=" * 50)

    passed = 0
    failed = 0
    skipped = 0

    for name, result in results.items():
        if result is True:
            status = "PASSED"
            passed += 1
        elif result is None:
            status = "SKIPPED"
            skipped += 1
        else:
            status = "FAILED"
            failed += 1
        logger.info(f"  {name}: {status}")

    logger.info("")
    logger.info(f"Total: {passed} passed, {failed} failed, {skipped} skipped")

    if failed == 0:
        logger.info("All tests passed!")
    else:
        logger.warning(f"{failed} test(s) failed")


if __name__ == "__main__":
    main()

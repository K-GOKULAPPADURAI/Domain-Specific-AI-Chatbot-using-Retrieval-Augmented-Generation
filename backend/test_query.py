#!/usr/bin/env python3
"""
Test script to verify that keyword and semantic search are working correctly.
Run this after uploading documents to test queries.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag_engine import get_rag_engine
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_query(question):
    """Test a single query."""
    print("\n" + "="*100)
    print(f"TEST QUERY: '{question}'")
    print("="*100)
    
    engine = get_rag_engine()
    
    try:
        answer = engine.query(question)
        print(f"\nANSWER:\n{answer}")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Test queries
    test_queries = [
        "Janssen 2017",
        "Gamez 2021",
        "What are the key findings?",
        "Soccer players ALS",
        "dementia athletes"
    ]
    
    if len(sys.argv) > 1:
        # Use custom query if provided
        test_query(" ".join(sys.argv[1:]))
    else:
        # Run predefined tests
        print("\n🧪 Running test queries...\n")
        for query in test_queries:
            test_query(query)
    
    print("\n✅ Tests complete!\n")

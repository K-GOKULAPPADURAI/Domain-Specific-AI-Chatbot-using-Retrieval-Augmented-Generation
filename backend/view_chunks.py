#!/usr/bin/env python3
"""
Utility script to view and analyze chunks created from uploaded documents.
Run this after uploading documents to see all chunks with detailed information.

Usage:
    python view_chunks.py              # Summary of all chunks
    python view_chunks.py 1            # View full content of chunk #1
    python view_chunks.py search TERM  # Search for TERM in all chunks
"""

import json
import os
from pathlib import Path

def print_chunks_summary():
    """Print summary of all chunks."""
    chunks_file = "chunks_debug.json"
    
    if not os.path.exists(chunks_file):
        print("❌ No chunks file found. Upload documents first and then run this script.")
        return
    
    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    print("\n" + "="*100)
    print("CHUNK ANALYSIS REPORT")
    print("="*100)
    
    # Summary statistics
    total_chunks = len(chunks)
    total_chars = sum(chunk["length"] for chunk in chunks)
    avg_length = total_chars // total_chunks if total_chunks > 0 else 0
    min_length = min(chunk["length"] for chunk in chunks) if chunks else 0
    max_length = max(chunk["length"] for chunk in chunks) if chunks else 0
    
    unique_sources = set()
    for chunk in chunks:
        unique_sources.add(chunk["source"])
    
    print(f"\n📊 STATISTICS:")
    print(f"   Total Chunks: {total_chunks}")
    print(f"   Total Characters: {total_chars:,}")
    print(f"   Average Chunk Size: {avg_length} characters")
    print(f"   Min Chunk Size: {min_length} characters")
    print(f"   Max Chunk Size: {max_length} characters")
    print(f"   Unique Sources: {len(unique_sources)}")
    
    if unique_sources:
        print(f"   Sources: {', '.join(unique_sources)}")
    
    # Chunks by source
    chunks_by_source = {}
    for chunk in chunks:
        source = chunk["source"]
        if source not in chunks_by_source:
            chunks_by_source[source] = []
        chunks_by_source[source].append(chunk)
    
    print(f"\n📁 CHUNKS BY SOURCE:")
    for source, source_chunks in chunks_by_source.items():
        source_chars = sum(c["length"] for c in source_chunks)
        print(f"   {source}: {len(source_chunks)} chunks ({source_chars:,} chars)")
    
    # Detailed chunks view
    print(f"\n📋 DETAILED CHUNK LISTING:")
    print("-"*100)
    
    for i, chunk in enumerate(chunks, 1):
        print(f"\n[CHUNK {i}]")
        print(f"Source: {chunk['source']} | Page: {chunk['page']} | Length: {chunk['length']} chars")
        print(f"Content Preview ({min(150, len(chunk['content']))} chars):")
        content_preview = chunk['content'][:150]
        if len(chunk['content']) > 150:
            content_preview += "..."
        print(f"  {content_preview}")
        print("-" * 100)

def print_chunk_details(chunk_id: int):
    """Print full details of a specific chunk."""
    chunks_file = "chunks_debug.json"
    
    if not os.path.exists(chunks_file):
        print("❌ No chunks file found.")
        return
    
    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    if chunk_id < 1 or chunk_id > len(chunks):
        print(f"❌ Invalid chunk ID. Valid range: 1-{len(chunks)}")
        return
    
    chunk = chunks[chunk_id - 1]
    
    print("\n" + "="*100)
    print(f"DETAILED VIEW - CHUNK {chunk_id}")
    print("="*100)
    print(f"Source: {chunk['source']}")
    print(f"Page: {chunk['page']}")
    print(f"Length: {chunk['length']} characters")
    print(f"\nFull Content:")
    print("-" * 100)
    print(chunk['content'])
    print("-" * 100)

def search_chunks(search_term: str):
    """Search for a term in all chunks and show matching results."""
    chunks_file = "chunks_debug.json"
    
    if not os.path.exists(chunks_file):
        print("❌ No chunks file found.")
        return
    
    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    print("\n" + "="*100)
    print(f"SEARCH RESULTS FOR: '{search_term}'")
    print("="*100)
    
    matches = []
    search_lower = search_term.lower()
    
    for i, chunk in enumerate(chunks, 1):
        content_lower = chunk['content'].lower()
        
        # Exact match
        if search_lower in content_lower:
            matches.append((i, chunk, 'exact'))
        # Fuzzy match (any word in search term)
        elif any(word in content_lower for word in search_term.lower().split()):
            matches.append((i, chunk, 'partial'))
    
    if not matches:
        print(f"\n❌ No matches found for '{search_term}'")
        return
    
    print(f"\n✅ Found {len(matches)} matching chunk(s):\n")
    
    for chunk_id, chunk, match_type in matches:
        match_indicator = "🎯 EXACT" if match_type == "exact" else "📌 PARTIAL"
        print(f"\n{match_indicator} MATCH - CHUNK {chunk_id}")
        print(f"Source: {chunk['source']} | Page: {chunk['page']}")
        
        # Highlight the search term in context
        content = chunk['content']
        start_pos = content.lower().find(search_lower)
        if start_pos >= 0:
            context_start = max(0, start_pos - 100)
            context_end = min(len(content), start_pos + len(search_term) + 100)
            preview = content[context_start:context_end]
            if context_start > 0:
                preview = "..." + preview
            if context_end < len(content):
                preview = preview + "..."
            print(f"Context: {preview}")
        else:
            # Partial match - show beginning
            print(f"Content: {content[:200]}...")
        
        print("-" * 100)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "search" and len(sys.argv) > 2:
            # Search for a term
            search_term = " ".join(sys.argv[2:])
            search_chunks(search_term)
        elif sys.argv[1].isdigit():
            # View specific chunk
            chunk_id = int(sys.argv[1])
            print_chunk_details(chunk_id)
        else:
            print("❌ Invalid argument")
            print("Usage:")
            print("  python view_chunks.py              # Summary")
            print("  python view_chunks.py 5            # View chunk #5")
            print("  python view_chunks.py search TERM  # Search for TERM")
    else:
        # View summary of all chunks
        print_chunks_summary()
    
    print("\n✅ Done!\n")

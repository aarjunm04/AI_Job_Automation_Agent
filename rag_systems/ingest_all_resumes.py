#!/usr/bin/env python3
# rag_systems/ingest_all_resumes.py

"""
Fresh ChromaDB Ingestion Script
- Deletes existing ChromaDB if present
- Creates fresh database
- Ingests all resumes with embeddings
- Includes retry logic for rate limits
"""

import sys
import os
import shutil
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_systems.resume_engine import get_default_engine

def clear_chromadb(chroma_path="./.chroma"):
    """Delete existing ChromaDB directory"""
    chroma_dir = Path(chroma_path)
    
    if chroma_dir.exists():
        print(f"\n🗑️  Found existing ChromaDB at: {chroma_dir}")
        print(f"   Deleting old database...")
        try:
            shutil.rmtree(chroma_dir)
            print(f"   ✅ Old ChromaDB deleted")
        except Exception as e:
            print(f"   ❌ Failed to delete: {e}")
            raise
    else:
        print(f"\n📁 No existing ChromaDB found at: {chroma_dir}")
    
    # Create fresh directory
    try:
        chroma_dir.mkdir(parents=True, exist_ok=True)
        print(f"   ✅ Created fresh ChromaDB directory")
    except Exception as e:
        print(f"   ❌ Failed to create directory: {e}")
        raise

def ingest_with_retry(engine, resume_id, max_retries=3):
    """Ingest a single resume with retry logic for rate limits"""
    for attempt in range(max_retries):
        try:
            result = engine.ingest_resume(resume_id)
            return result, None
        except Exception as e:
            error_str = str(e)
            # Check for rate limit (429) error
            if '429' in error_str and attempt < max_retries - 1:
                wait_time = (attempt + 1) * 30  # 30s, 60s, 90s
                print(f"⏳ Rate limit, waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                return None, error_str
    return None, "Max retries exceeded"

def verify_ingestion(chroma_path="./.chroma"):
    """Verify ChromaDB was populated correctly"""
    try:
        from chromadb import PersistentClient
        client = PersistentClient(path=chroma_path)
        collection = client.get_collection("resumes")
        count = collection.count()
        
        results = collection.get(limit=100, include=['metadatas'])
        anchors = sum(1 for m in results['metadatas'] if m.get('anchor') == True)
        chunks = count - anchors
        
        return {
            'total': count,
            'anchors': anchors,
            'chunks': chunks,
            'success': count > 0
        }
    except Exception as e:
        return {
            'total': 0,
            'anchors': 0,
            'chunks': 0,
            'success': False,
            'error': str(e)
        }

def main():
    print("█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + "  FRESH CHROMADB INGESTION - ALL RESUMES".center(78) + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)
    
    try:
        # Step 1: Clear existing ChromaDB
        print("\n" + "=" * 80)
        print("STEP 1: CLEARING EXISTING CHROMADB")
        print("=" * 80)
        clear_chromadb()
        
        # Step 2: Initialize engine
        print("\n" + "=" * 80)
        print("STEP 2: INITIALIZING RAG ENGINE")
        print("=" * 80)
        print("\n⚙️  Loading resume configuration and embedder...")
        engine = get_default_engine()
        total = len(engine.resumes)
        print(f"✅ Loaded {total} resumes from config")
        
        # Step 3: Ingest all resumes
        print("\n" + "=" * 80)
        print("STEP 3: INGESTING ALL RESUMES")
        print("=" * 80)
        print()
        
        success_count = 0
        failed = []
        
        for idx, resume_id in enumerate(engine.resumes.keys(), 1):
            print(f"[{idx:2d}/{total}] Ingesting {resume_id:35}... ", end="", flush=True)
            
            result, error = ingest_with_retry(engine, resume_id)
            
            if result:
                print(f"✅ {result['num_chunks']} chunks")
                success_count += 1
            else:
                print(f"❌ FAILED: {error}")
                failed.append((resume_id, error))
        
        # Step 4: Verify ingestion
        print("\n" + "=" * 80)
        print("STEP 4: VERIFYING CHROMADB")
        print("=" * 80)
        
        verification = verify_ingestion()
        
        if verification['success']:
            print(f"\n✅ ChromaDB populated successfully:")
            print(f"   Total embeddings: {verification['total']}")
            print(f"   - Anchor vectors: {verification['anchors']}")
            print(f"   - Chunk vectors:  {verification['chunks']}")
            print(f"   - Chunks per resume: {verification['chunks'] / total:.1f}")
        else:
            print(f"\n⚠️  ChromaDB verification issue:")
            print(f"   {verification.get('error', 'Unknown error')}")
        
        # Final Summary
        print("\n" + "=" * 80)
        print("INGESTION SUMMARY")
        print("=" * 80)
        
        success_rate = (success_count / total * 100) if total > 0 else 0
        print(f"\n  Successful: {success_count}/{total} ({success_rate:.1f}%)")
        
        if failed:
            print(f"\n  ❌ Failed resumes ({len(failed)}):")
            for resume_id, error in failed:
                print(f"     - {resume_id}")
                print(f"       Error: {error[:100]}...")
        else:
            print(f"\n  🎉 All resumes ingested successfully!")
        
        print("\n" + "=" * 80)
        
        # Exit code
        if success_count == total:
            print("\n✅ INGESTION COMPLETE - System ready for production\n")
            sys.exit(0)
        elif success_count > 0:
            print("\n⚠️  INGESTION PARTIAL - Review failed resumes\n")
            sys.exit(0)
        else:
            print("\n❌ INGESTION FAILED - No resumes ingested\n")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

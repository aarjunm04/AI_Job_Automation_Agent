# rag_systems/ingest_all_resumes.py

"""
One-time script to ingest all resumes into ChromaDB
Run this before starting the production server for the first time
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_systems.resume_engine import get_default_engine

def main():
    print("=" * 80)
    print("INGESTING ALL RESUMES INTO CHROMADB")
    print("=" * 80)
    
    try:
        engine = get_default_engine()
        
        total = len(engine.resumes)
        print(f"\nFound {total} resumes to ingest\n")
        
        success_count = 0
        failed = []
        
        for idx, resume_id in enumerate(engine.resumes.keys(), 1):
            print(f"[{idx}/{total}] Ingesting {resume_id}...", end=" ")
            
            try:
                result = engine.ingest_resume(resume_id)
                print(f"✓ {result['num_chunks']} chunks")
                success_count += 1
            except Exception as e:
                print(f"✗ FAILED: {e}")
                failed.append((resume_id, str(e)))
        
        print("\n" + "=" * 80)
        print(f"INGESTION COMPLETE: {success_count}/{total} successful")
        
        if failed:
            print(f"\nFailed resumes ({len(failed)}):")
            for resume_id, error in failed:
                print(f"  - {resume_id}: {error}")
        
        print("=" * 80)
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

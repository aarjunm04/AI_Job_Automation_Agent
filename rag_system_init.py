#!/usr/bin/env python3
"""
RAG SYSTEM INITIALIZER / DIAGNOSTICS SCRIPT (rag_systemS VERSION)

Place this file in the ROOT of your project:

AI_Job_Automation_Agent/
    rag_systems/
    rag_systems_init.py  <--- HERE

Run:
    python rag_systems_init.py

This script will:
1. Fix PYTHONPATH automatically
2. Import rag_systems modules correctly
3. Validate Gemini API Key
4. Run healthcheck
5. Ingest all resumes (PDF -> text -> chunks -> embeddings -> Chroma)
6. Perform test resume selection
"""

import os
import sys
import json
import traceback

print("\n🚀 RAG SYSTEM INITIALIZATION SCRIPT (rag_systems version)\n")

# ---------------------------------------------------------
# STEP 1 — Ensure correct project root
# ---------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR

print(f"📌 Project root detected as: {PROJECT_ROOT}")

rag_systems_path = os.path.join(PROJECT_ROOT, "rag_systems")
if not os.path.isdir(rag_systems_path):
    print("❌ ERROR: rag_systems/ directory not found in this folder.")
    print("Make sure this file is placed at your project root.")
    sys.exit(1)

# ---------------------------------------------------------
# STEP 2 — Fix PYTHONPATH
# ---------------------------------------------------------

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    print("🔧 PYTHONPATH updated with project root.\n")

# ---------------------------------------------------------
# STEP 3 — Validate Gemini API Key
# ---------------------------------------------------------

GEMINI_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_KEY:
    print("❌ ERROR: GEMINI_API_KEY is not set.")
    print("Set it via:")
    print('   export GEMINI_API_KEY="your_key_here"')
    sys.exit(1)
else:
    print("🔑 Gemini API Key found.\n")

# ---------------------------------------------------------
# STEP 4 — Import rag_systems modules
# ---------------------------------------------------------

try:
    from rag_systems.rag_api import healthcheck, get_resume_pdf_path, select_resume, get_rag_context
    from rag_systems.resume_engine import get_default_engine
except Exception:
    print("❌ ERROR: Could not import rag_systems modules.\n")
    traceback.print_exc()
    sys.exit(1)

print("📦 rag_systems successfully imported.\n")

# ---------------------------------------------------------
# STEP 5 — Initialize Resume Engine
# ---------------------------------------------------------

try:
    engine = get_default_engine()
    print("⚙️ ResumeEngine initialized.\n")
except Exception:
    print("❌ ERROR: Failed to initialize ResumeEngine.")
    traceback.print_exc()
    sys.exit(1)

# ---------------------------------------------------------
# STEP 6 — Healthcheck
# ---------------------------------------------------------

print("🩺 Running RAG healthcheck...\n")

try:
    status = healthcheck()
    print("Healthcheck result:")
    print(json.dumps(status, indent=2), "\n")

    if status.get("status") != "ok":
        print("❌ Healthcheck failed. Fix above issues.")
        sys.exit(1)

except Exception:
    print("❌ ERROR during healthcheck.")
    traceback.print_exc()
    sys.exit(1)

# ---------------------------------------------------------
# STEP 7 — Ingest all resumes
# ---------------------------------------------------------

print("📥 Ingesting all resumes from resume_config.json...\n")

try:
    resumes = engine.list_resumes()
    if not resumes:
        print("❌ No resumes found in config file.")
        sys.exit(1)

    for r in resumes:
        rid = r["resume_id"]
        print(f"➡️ Ingesting: {rid}")
        result = engine.ingest_resume(rid)
        print(f"   ✔️ {result['num_chunks']} chunks stored")

    print("\n🎉 All resumes ingested successfully!\n")

except Exception:
    print("❌ ERROR during resume ingestion.")
    traceback.print_exc()
    sys.exit(1)

# ---------------------------------------------------------
# STEP 8 — Quick Resume Selection Test
# ---------------------------------------------------------

print("🧪 Running resume selection smoke test...\n")

demo_job = """
We need an AI/ML engineer with Python, Pytorch, LLMs, automation/agents experience, 
and vector embeddings or workflow orchestration background.
"""

try:
    result = select_resume({"job_text": demo_job})

    print("Resume selection output:")
    print(json.dumps({
        "top_resume_id": result["top_resume_id"],
        "top_score": result["top_score"]
    }, indent=2))

    print("\n📄 Selected resume path:")
    print(get_resume_pdf_path(result["top_resume_id"]))

except Exception:
    print("❌ ERROR during resume selection test.")
    traceback.print_exc()
    sys.exit(1)

# ---------------------------------------------------------
# DONE
# ---------------------------------------------------------

print("\n✅ RAG SYSTEM FULLY INITIALIZED & VERIFIED")
print("You are now ready for MCP integration.\n")

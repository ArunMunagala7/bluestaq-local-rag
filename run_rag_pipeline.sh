#!/bin/zsh
# run_rag_pipeline.sh - Complete end-to-end RAG automation
# This script handles setup, ingestion, querying, and evaluation

set -e  # Exit on error

echo "════════════════════════════════════════════════════════════════"
echo "  🧠 Bluestaq Local RAG - Automated Pipeline"
echo "════════════════════════════════════════════════════════════════\n"

# ──────────────────────────────────────────────────────────────────
# 1. ENVIRONMENT SETUP
# ──────────────────────────────────────────────────────────────────
echo "📦 Step 1: Checking environment..."

if [ ! -d "venv" ]; then
    echo "   Creating virtual environment..."
    python3 -m venv venv
    echo "   ✅ Virtual environment created"
else
    echo "   ✅ Virtual environment exists"
fi

source venv/bin/activate
echo "   ✅ Virtual environment activated"

# ──────────────────────────────────────────────────────────────────
# 2. DEPENDENCY INSTALLATION
# ──────────────────────────────────────────────────────────────────
echo "\n📦 Step 2: Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
echo "   ✅ Dependencies installed"

# ──────────────────────────────────────────────────────────────────
# 3. MODEL VERIFICATION
# ──────────────────────────────────────────────────────────────────
echo "\n🧠 Step 3: Verifying model file..."
if [ -f "models/gguf/Llama-3.2-3B-Instruct-Q4_K_M.gguf" ]; then
    MODEL_SIZE=$(du -h "models/gguf/Llama-3.2-3B-Instruct-Q4_K_M.gguf" | cut -f1)
    echo "   ✅ Model found (Size: $MODEL_SIZE)"
else
    echo "   ❌ Model not found!"
    echo "   Please download to: models/gguf/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
    exit 1
fi

# ──────────────────────────────────────────────────────────────────
# 4. CORPUS INGESTION
# ──────────────────────────────────────────────────────────────────
echo "\n📚 Step 4: Processing corpus..."

# Check if uploads directory has new files
UPLOAD_COUNT=$(find data/uploads -type f 2>/dev/null | wc -l | tr -d ' ')

if [ "$UPLOAD_COUNT" -gt 0 ]; then
    echo "   Found $UPLOAD_COUNT file(s) in uploads directory"
    echo "   Running bulk upload..."
    python -m app.app bulk-upload
    echo "   ✅ Corpus updated and indexed"
else
    if [ -f "data/index/faiss.index" ]; then
        CHUNK_COUNT=$(find data/corpus -name "*.txt" -type f 2>/dev/null | wc -l | tr -d ' ')
        echo "   ✅ FAISS index exists ($CHUNK_COUNT documents)"
    else
        echo "   Building initial FAISS index..."
        python -m app.ingest
        echo "   ✅ FAISS index created"
    fi
fi

# ──────────────────────────────────────────────────────────────────
# 5. EVALUATION QUERIES
# ──────────────────────────────────────────────────────────────────
echo "\n🧪 Step 5: Running evaluation queries...\n"

echo "════════════════════════════════════════════════════════════════"
echo "Test 1: Grounding (Concise Style)"
echo "════════════════════════════════════════════════════════════════"
python -m app.app query-rag "What were the three estates in French society before 1789?" --style concise

echo "\n════════════════════════════════════════════════════════════════"
echo "Test 2: Citation Tracking (Detailed + Justification)"
echo "════════════════════════════════════════════════════════════════"
python -m app.app query-rag "How did Columbus treat the Arawak Indians when he first arrived?" --style detailed --justify

echo "\n════════════════════════════════════════════════════════════════"
echo "Test 3: Absence Detection"
echo "════════════════════════════════════════════════════════════════"
python -m app.app query-rag "What does the corpus say about the American Revolution?" --style concise

echo "\n════════════════════════════════════════════════════════════════"
echo "Test 4: Style Adherence (Bullet Points)"
echo "════════════════════════════════════════════════════════════════"
python -m app.app query-rag "What were the main causes of the French Revolution?" --style bullet

# ──────────────────────────────────────────────────────────────────
# 6. COMPLETION
# ──────────────────────────────────────────────────────────────────
echo "\n════════════════════════════════════════════════════════════════"
echo "  ✨ Pipeline complete!"
echo "════════════════════════════════════════════════════════════════\n"

echo "Next steps:"
echo "  • Interactive chat:  python -m app.app chat"
echo "  • Custom query:      python -m app.app query-rag \"<your question>\" --style detailed"
echo "  • Add documents:     Copy files to data/uploads/ then run: python -m app.app bulk-upload"
echo ""

#!/bin/zsh
# setup.sh - Automated environment setup and initialization

echo "🚀 Setting up Local RAG environment..."

# 1. Create Python virtual environment
python3 -m venv venv
echo "✅ Virtual environment created"

# 2. Activate virtual environment
source venv/bin/activate
echo "✅ Virtual environment activated"

# 3. Upgrade pip
pip install --upgrade pip
echo "✅ pip upgraded"

# 4. Install dependencies
pip install -r requirements.txt
echo "✅ Dependencies installed"

# 5. Verify model exists
if [ -f "models/gguf/Llama-3.2-3B-Instruct-Q4_K_M.gguf" ]; then
    echo "✅ Model file found"
else
    echo "⚠️  Model file not found. Please download Llama-3.2-3B-Instruct-Q4_K_M.gguf to models/gguf/"
fi

# 6. Check if FAISS index exists, rebuild if missing
if [ -f "data/index/faiss.index" ]; then
    echo "✅ FAISS index found"
else
    echo "🔨 Building FAISS index from corpus..."
    python -m app.ingest
    echo "✅ FAISS index built"
fi

# 7. Run a test query
echo "\n🧪 Running test query..."
python -m app.app query-rag "What were the three estates in French society before 1789?" --style concise

echo "\n✨ Setup complete! Use 'source venv/bin/activate' to activate the environment."

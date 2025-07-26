#!/bin/bash

# Brain Tumor Detection - Quick Start Script
# =========================================

echo "🧠 Brain Tumor Detection with Vision Transformer"
echo "================================================="

# Check if Python environment is activated
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment is active: $VIRTUAL_ENV"
else
    echo "⚠️  No virtual environment detected. Consider activating one."
fi

# Install dependencies if requirements.txt exists
if [ -f "requirements.txt" ]; then
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
else
    echo "❌ requirements.txt not found"
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p models
mkdir -p data
mkdir -p uploads
mkdir -p logs

# Function to run training
run_training() {
    echo "🚀 Starting model training..."
    python train_model.py \
        --epochs 50 \
        --batch_size 32 \
        --lr 1e-4 \
        --model_name vit_base_patch16_224 \
        --experiment_name brain_tumor_vit_$(date +%Y%m%d_%H%M%S)
}

# Function to run API server
run_api() {
    echo "🌐 Starting API server..."
    cd api
    python main.py
}

# Function to run Gradio app
run_gradio() {
    echo "🎨 Starting Gradio interface..."
    cd api
    python gradio_app.py
}

# Function to run Jupyter notebook
run_notebook() {
    echo "📓 Starting Jupyter notebook..."
    jupyter notebook
}

# Main menu
echo ""
echo "Select an option:"
echo "1) Train model"
echo "2) Run API server"
echo "3) Run Gradio interface"
echo "4) Open Jupyter notebooks"
echo "5) Install dependencies only"
echo "6) Exit"

read -p "Enter your choice (1-6): " choice

case $choice in
    1)
        run_training
        ;;
    2)
        run_api
        ;;
    3)
        run_gradio
        ;;
    4)
        run_notebook
        ;;
    5)
        echo "✅ Dependencies installed"
        ;;
    6)
        echo "👋 Goodbye!"
        exit 0
        ;;
    *)
        echo "❌ Invalid option"
        exit 1
        ;;
esac

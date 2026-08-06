#!/bin/bash
set -e

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate

echo "Installing f1-tracking (editable) with dev + app extras..."
pip install --upgrade pip -q
pip install -e ".[dev,app]" -q

python -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT FOUND (CPU mode)')"

echo ""
echo "Setup complete. Activate with: source .venv/bin/activate"
echo "Try: f1-track --help"

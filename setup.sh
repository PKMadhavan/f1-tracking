#!/bin/bash
pip install -q -r requirements.txt
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT FOUND')"
echo "Setup complete."

python -m venv tortoise-venv
source ./tortoise-venv/bin/activate
python -m pip install --upgrade pip
# CUDA
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
python -m pip install -r ./requirements.txt
deactivate

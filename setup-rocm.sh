python -m venv tortoise-venv
source ./tortoise-venv/bin/activate
python -m pip install --upgrade pip
# ROCM
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/rocm5.1.1 # 5.2 does not work for me desu
python -m pip install -r ./requirements.txt
python setup.py install
deactivate

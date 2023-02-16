python -m venv tortoise-venv
call .\tortoise-venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio torch-directml
python -m pip install -r ./requirements.txt
python -m pip install -r ./requirements_legacy.txt
deactivate
pause
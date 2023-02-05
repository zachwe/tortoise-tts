git pull
python -m venv tortoise-venv
call .\tortoise-venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r ./requirements.txt
deactivate
pause
call .\tortoise-venv\Scripts\activate.bat
accelerate launch --num_cpu_threads_per_process=6 app.py
deactivate
pause
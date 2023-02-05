@echo off
rm .\in\.gitkeep
rm .\out\.gitkeep
for %%a in (".\in\*.*") do ffmpeg -i "%%a" -ar 22050 -ac 1 -c:a pcm_f32le ".\out\%%~na.wav"
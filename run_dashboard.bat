@echo off
title Pathology Depth Analysis - Presentation Dashboard
echo ===================================================================
echo  NeuroPath DepthLab: Histopathology AI Depth Degradation Dashboard
echo ===================================================================
echo.
echo Starting Flask presentation dashboard server on http://127.0.0.1:5000 ...
echo.

:: Launch default web browser after 1.5 seconds
start "" timeout /t 2 /nobreak >nul & start http://127.0.0.1:5000

:: Start Python backend
python app.py

pause

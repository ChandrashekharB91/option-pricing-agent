@echo off
rem Daily feed for Windows Task Scheduler: runs from the repo folder with the project virtual environment.
cd /d "%~dp0.."
if not exist data\feed mkdir data\feed
".venv\Scripts\python.exe" scripts\daily_feed.py --send >> data\feed\task.log 2>&1

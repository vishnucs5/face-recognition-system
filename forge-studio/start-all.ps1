# Forge Studio Complete Workstation Launcher
# Launches ComfyUI, Ollama (GPU-accelerated), Backend, and Frontend

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  Starting Forge Studio Workstation Stack" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# 1. Start Ollama with GPU acceleration
Write-Host "[1/4] Starting Ollama with RTX 5070 Ti GPU Acceleration..." -ForegroundColor Green
Remove-Item Env:OLLAMA_LLM_LIBRARY -ErrorAction SilentlyContinue
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Remove-Item Env:OLLAMA_LLM_LIBRARY -ErrorAction SilentlyContinue; ollama serve" -WindowStyle Minimized

# 2. Start ComfyUI
Write-Host "[2/4] Starting ComfyUI on Port 8188..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "C:\Users\pc\AppData\Local\Programs\Python\Python311\python.exe c:\vishnu\ComfyUI\main.py --listen 127.0.0.1 --port 8188 --cpu" -WindowStyle Minimized

Start-Sleep -Seconds 3

# 3. Start Backend
Write-Host "[3/4] Starting Forge Studio Backend on Port 8000..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd c:\vishnu\forge-studio\backend; `$env:PYTHONPATH='.'; c:\vishnu\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000" -WindowStyle Minimized

Start-Sleep -Seconds 2

# 4. Start Frontend
Write-Host "[4/4] Starting Forge Studio Frontend on Port 5173..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd c:\vishnu\forge-studio\frontend; npm run dev -- --host 127.0.0.1 --port 5173" -WindowStyle Minimized

Write-Host ""
Write-Host "All services launched!" -ForegroundColor Green
Write-Host "Studio UI: http://127.0.0.1:5173" -ForegroundColor Yellow
Write-Host "API docs:  http://127.0.0.1:8000/docs" -ForegroundColor Yellow
Write-Host "ComfyUI:   http://127.0.0.1:8188" -ForegroundColor Yellow
Write-Host "Ollama:    http://127.0.0.1:11434" -ForegroundColor Yellow

# Forge Studio Workstation Shutdown Script
# Gracefully stops ComfyUI, Ollama, Backend, and Frontend

Write-Host "Stopping Forge Studio Workstation processes..." -ForegroundColor Yellow

Get-Process -Name "uvicorn", "node", "ollama", "llama-server" -ErrorAction SilentlyContinue | ForEach-Object {
    Write-Host "Stopping $($_.ProcessName) (PID $($_.Id))..."
    Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
}

Write-Host "Done. All services stopped." -ForegroundColor Green

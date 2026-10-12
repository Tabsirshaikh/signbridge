# SignBridge: backend in a new window + frontend in this terminal (both start right away).
# First model load can take 1-2 minutes. Vite starts at http://localhost:3000 right away.
# Put your Gemini key in backend\.env (see README).
$Root = $PSScriptRoot
$Backend = Join-Path $Root "backend"
$Frontend = Join-Path $Root "frontend"

if (-not (Test-Path (Join-Path $Backend "app.py"))) {
    Write-Error "backend\app.py not found under $Root"
    exit 1
}

Write-Host "Opening backend in a new window (leave it open - first load is slow)..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "Set-Location -LiteralPath '$Backend'; Write-Host 'Starting Flask on http://127.0.0.1:5000 ...' -ForegroundColor Cyan; python app.py"
)

Write-Host ""
Write-Host "Starting Vite on http://localhost:3000 (does not wait for the backend)." -ForegroundColor Green
Write-Host "If localhost fails, try http://127.0.0.1:3000" -ForegroundColor DarkGray
Write-Host ""

Set-Location -LiteralPath $Frontend
npm run dev

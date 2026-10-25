# run_all.ps1 - Automated Setup, Test, Validation, and Dashboard Launch Script
# Operating System: Windows PowerShell

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "    CHILL2GRID AI - AUTOMATED DEPLOYMENT & VALIDATION    " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Environment Verification
$pythonCmd = "python"
if (-not (Get-Command $pythonCmd -ErrorAction SilentlyContinue)) {
    Write-Error "Python executable not found in PATH. Please install Python 3.11+."
    exit 1
}

# 2. Virtual Environment (Optional / Local)
if (-not (Test-Path ".venv")) {
    Write-Host "`n[1/5] Creating Python virtual environment (.venv)..." -ForegroundColor Yellow
    & $pythonCmd -m venv .venv
}
if (Test-Path ".venv\Scripts\Activate.ps1") {
    Write-Host "[1/5] Activating virtual environment..." -ForegroundColor Yellow
    . .venv\Scripts\Activate.ps1
}

# 3. Install Dependencies
Write-Host "`n[2/5] Installing pinned dependencies from requirements.txt..." -ForegroundColor Yellow
& python -m pip install -r requirements.txt --quiet

# 4. Execute Full Pytest Test Suite
Write-Host "`n[3/5] Running pytest test suite..." -ForegroundColor Yellow
& python -m pytest
if ($LASTEXITCODE -ne 0) {
    Write-Error "Pytest suite failed. Halting deployment."
    exit 1
}
Write-Host "[3/5] All unit and integration tests passed!" -ForegroundColor Green

# 5. Run Self-Check
Write-Host "`n[4/5] Running end-to-end integration self-check..." -ForegroundColor Yellow
& python -m app.selfcheck
if ($LASTEXITCODE -ne 0) {
    Write-Error "Self-check failed."
    exit 1
}

# 6. Start Streamlit Dashboard
Write-Host "`n[5/5] Launching Chill2Grid AI Streamlit Dashboard on port 8501..." -ForegroundColor Green
Write-Host "Access dashboard at: http://localhost:8501" -ForegroundColor Cyan
& python -m streamlit run app/dashboard.py

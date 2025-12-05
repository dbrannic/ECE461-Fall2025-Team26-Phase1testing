# ML Model Evaluation System - PowerShell Run Script
# Usage: .\run.ps1 [install|test|URL_FILE]

param(
    [string]$Command = ""
)

# Helper functions for colored output
function Write-Success {
    param([string]$Message)
    Write-Host "✓ $Message" -ForegroundColor Green
}

function Write-ErrorMsg {
    param([string]$Message)
    Write-Host "✗ Error: $Message" -ForegroundColor Red
}

function Write-Warning {
    param([string]$Message)
    Write-Host "⚠ Warning: $Message" -ForegroundColor Yellow
}

function Write-Info {
    param([string]$Message)
    Write-Host "ℹ $Message" -ForegroundColor Blue
}

function Install-Dependencies {
    Write-Host "Installing ML Model Evaluation System dependencies..." -ForegroundColor Blue

    # Check if Python is available
    $pythonCmd = $null
    
    try {
        $null = py --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            $pythonCmd = "py"
            Write-Host "Using Python command: py" -ForegroundColor Blue
        }
    } catch { }
    
    if (-not $pythonCmd) {
        try {
            $null = python --version 2>&1
            if ($LASTEXITCODE -eq 0) {
                $pythonCmd = "python"
                Write-Host "Using Python command: python" -ForegroundColor Blue
            }
        } catch { }
    }
    
    if (-not $pythonCmd) {
        Write-ErrorMsg "Python is not installed or not in PATH"
        exit 1
    }

    # Check if pip is available
    try {
        $null = & $pythonCmd -m pip --version 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-ErrorMsg "pip is not available. Please install pip first."
            exit 1
        }
    } catch {
        Write-ErrorMsg "pip is not available. Please install pip first."
        exit 1
    }

    # Install requirements
    if (Test-Path "requirements.txt") {
        Write-Host "Installing packages from requirements.txt..." -ForegroundColor Blue
        
        & $pythonCmd -m pip install -r requirements.txt --user 2>&1 | Out-Null
        
        if ($LASTEXITCODE -eq 0) {
            Write-Success "All dependencies installed successfully!"
            exit 0
        } else {
            Write-ErrorMsg "Failed to install dependencies"
            exit 1
        }
    } else {
        Write-ErrorMsg "requirements.txt not found in the current directory"
        exit 1
    }
}

function Run-Tests {
    # Check if Python is available
    $pythonCmd = $null
    
    try {
        $null = py --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            $pythonCmd = "py"
        }
    } catch { }
    
    if (-not $pythonCmd) {
        try {
            $null = python --version 2>&1
            if ($LASTEXITCODE -eq 0) {
                $pythonCmd = "python"
            }
        } catch { }
    }
    
    if (-not $pythonCmd) {
        Write-Host "Error: Python is not installed" -ForegroundColor Red
        exit 1
    }

    # Check if pytest is installed
    try {
        $null = & $pythonCmd -m pytest --version 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Error: pytest is not installed. Run '.\run.ps1 install' first." -ForegroundColor Red
            exit 1
        }
    } catch {
        Write-Host "Error: pytest is not installed. Run '.\run.ps1 install' first." -ForegroundColor Red
        exit 1
    }

    # Run pytest with coverage and capture output
    $pytestOutput = & $pythonCmd -m pytest backend\src\Testing\ `
        --cov=backend\src `
        --cov-report=term `
        --tb=line `
        -v 2>&1 | Out-String
    
    $testExitCode = $LASTEXITCODE
    
    # Print output to stderr for debugging
    Write-Host $pytestOutput -ForegroundColor Gray
    
    # === PARSE TEST COUNTS ===
    $passed = 0
    $failed = 0
    
    # Look for: "187 passed in 2.34s"
    if ($pytestOutput -match '(\d+) passed') {
        $passed = [int]$Matches[1]
    }
    
    if ($pytestOutput -match '(\d+) failed') {
        $failed = [int]$Matches[1]
    }
    
    $total = $passed + $failed
    
    # Fallback: count PASSED/FAILED markers
    if ($total -eq 0) {
        $passed = ([regex]::Matches($pytestOutput, "PASSED")).Count
        $failed = ([regex]::Matches($pytestOutput, "FAILED")).Count
        $total = $passed + $failed
    }
    
    # === PARSE COVERAGE ===
    $coverage = 0
    
    # Look for TOTAL line
    if ($pytestOutput -match 'TOTAL.*?(\d+)%') {
        $coverage = [int]$Matches[1]
    }
    
    # Ensure valid defaults
    if ($passed -eq $null) { $passed = 0 }
    if ($total -eq $null) { $total = 0 }
    if ($coverage -eq $null) { $coverage = 0 }
    
    # === CRITICAL: USE Write-Output (NOT Write-Host) ===
    # Write-Output goes to stdout, Write-Host does not!
    Write-Output "${passed}/${total} test cases passed. ${coverage}% line coverage achieved."
    
    # Exit with pytest's exit code
    exit $testExitCode
}

function Run-Evaluation {
    param([string]$UrlFile)

    if (-not (Test-Path $UrlFile)) {
        Write-Host "Error: URL file '$UrlFile' not found" -ForegroundColor Red
        exit 1
    }

    # Check if Python is available
    $pythonCmd = $null
    
    try {
        $null = python --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            $pythonCmd = "python"
        }
    } catch { }
    
    if (-not $pythonCmd) {
        try {
            $null = py --version 2>&1
            if ($LASTEXITCODE -eq 0) {
                $pythonCmd = "py"
            }
        } catch { }
    }
    
    if (-not $pythonCmd) {
        Write-Host "Error: Python is not installed" -ForegroundColor Red
        exit 1
    }

    Push-Location backend
    try {
        & $pythonCmd src\main.py $UrlFile
        $evalExitCode = $LASTEXITCODE
        exit $evalExitCode
    } finally {
        Pop-Location
    }
}

function Show-Usage {
    Write-Host "ML Model Evaluation System" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Usage: .\run.ps1 [COMMAND|URL_FILE]" -ForegroundColor White
    Write-Host ""
    Write-Host "Commands:" -ForegroundColor Yellow
    Write-Host "  install     Install all required dependencies" -ForegroundColor White
    Write-Host "  test        Run the test suite" -ForegroundColor White
    Write-Host "  URL_FILE    Run evaluation on URLs in the specified file" -ForegroundColor White
    Write-Host ""
}

# Main script logic
if (-not $Command) {
    Write-Host "Error: No command or URL file specified" -ForegroundColor Red
    Show-Usage
    exit 1
}

switch ($Command.ToLower()) {
    "install" {
        Install-Dependencies
    }
    "test" {
        Run-Tests
    }
    { $_ -in @("-h", "--help", "help") } {
        Show-Usage
        exit 0
    }
    default {
        # Assume it's a URL file
        Run-Evaluation $Command
    }
}
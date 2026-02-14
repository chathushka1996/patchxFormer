# Quick LaTeX Installation Checker
Write-Host "Checking for LaTeX installation..." -ForegroundColor Cyan
Write-Host ""

# Check for latexmk
$latexmk = Get-Command latexmk -ErrorAction SilentlyContinue
if ($latexmk) {
    Write-Host "✓ latexmk found at: $($latexmk.Source)" -ForegroundColor Green
    Write-Host "  Version: " -NoNewline
    & latexmk --version 2>&1 | Select-Object -First 1
} else {
    Write-Host "✗ latexmk NOT FOUND" -ForegroundColor Red
}

Write-Host ""

# Check for pdflatex
$pdflatex = Get-Command pdflatex -ErrorAction SilentlyContinue
if ($pdflatex) {
    Write-Host "✓ pdflatex found at: $($pdflatex.Source)" -ForegroundColor Green
} else {
    Write-Host "✗ pdflatex NOT FOUND" -ForegroundColor Red
}

Write-Host ""

# Check common installation locations
Write-Host "Checking common LaTeX installation locations..." -ForegroundColor Cyan
$locations = @(
    "$env:LOCALAPPDATA\Programs\MiKTeX",
    "$env:ProgramFiles\MiKTeX",
    "${env:ProgramFiles(x86)}\MiKTeX",
    "C:\texlive",
    "C:\Program Files\MiKTeX"
)

$found = $false
foreach ($loc in $locations) {
    if (Test-Path $loc) {
        Write-Host "✓ Found: $loc" -ForegroundColor Green
        $found = $true
        
        # Check for bin directory
        $binDirs = Get-ChildItem -Path $loc -Recurse -Directory -Filter "bin" -ErrorAction SilentlyContinue | Select-Object -First 3
        foreach ($binDir in $binDirs) {
            $latexmkPath = Join-Path $binDir.FullName "latexmk.exe"
            if (Test-Path $latexmkPath) {
                Write-Host "  → latexmk.exe found at: $latexmkPath" -ForegroundColor Yellow
                Write-Host "  → Add this directory to PATH: $($binDir.FullName)" -ForegroundColor Yellow
            }
        }
    }
}

if (-not $found) {
    Write-Host "✗ No LaTeX installation found in common locations" -ForegroundColor Red
    Write-Host ""
    Write-Host "INSTALLATION REQUIRED:" -ForegroundColor Yellow
    Write-Host "1. Download MiKTeX from: https://miktex.org/download" -ForegroundColor White
    Write-Host "2. Install and check 'Add MiKTeX to PATH'" -ForegroundColor White
    Write-Host "3. Restart VS Code/Cursor" -ForegroundColor White
    Write-Host ""
    Write-Host "Or use Overleaf (online, no installation): https://www.overleaf.com/" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "Current PATH entries related to LaTeX:" -ForegroundColor Cyan
$env:Path -split ';' | Where-Object { $_ -match 'miktex|texlive|latex' } | ForEach-Object {
    Write-Host "  → $_" -ForegroundColor Gray
}
if (-not ($env:Path -match 'miktex|texlive|latex')) {
    Write-Host "  (none found)" -ForegroundColor Gray
}


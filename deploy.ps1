# nvyra-x deployment and benchmark script (powershell)

param(
    [Parameter(Position=0)]
    [string]$Command,
    
    [Parameter(Position=1)]
    [string]$InputFile
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"

# HARDCODED MODAL CREDENTIALS
$env:MODAL_TOKEN_ID = "ak-a7u4uUkwIMei48oRWRL3Ao"
$env:MODAL_TOKEN_SECRET = "as-ZZjpCIrClYEVghEDPu8Zr9"
Write-Host "Using Hardcoded Modal Credentials..." -ForegroundColor Magenta

Write-Host "nvyra-x sota deployment script" -ForegroundColor Cyan
Write-Host "===============================" -ForegroundColor Cyan
Write-Host ""

# load environment variables from .env
$envFile = ".env"
if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        if ($_ -match "^([^#][^=]+)=(.*)$") {
            [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2])
        }
    }
}

switch ($Command) {
    "deploy-free" {
        Write-Host "deploying free tier (cpu) to modal..." -ForegroundColor Green
        Push-Location nvyra-x-free
        modal deploy inference.py
        Pop-Location
        Write-Host "done! endpoint deployed." -ForegroundColor Green
    }
    
    "deploy-pro" {
        Write-Host "deploying pro tier (h200) to modal..." -ForegroundColor Green
        Push-Location nvyra-x-pro
        modal deploy inference.py
        Pop-Location
        Write-Host "done! endpoint deployed." -ForegroundColor Green
    }

    "deploy-storage" {
        Write-Host "deploying storage worker (h200) to modal..." -ForegroundColor Green
        Push-Location nvyra-x-pro
        modal deploy storage_worker.py
        Pop-Location
        Write-Host "done! worker deployed." -ForegroundColor Green
    }
    
    "deploy-all" {
        Write-Host "deploying all services to modal..." -ForegroundColor Green
        Push-Location nvyra-x-free
        modal deploy inference.py
        Pop-Location
        Push-Location nvyra-x-pro
        modal deploy inference.py
        modal deploy storage_worker.py
        Pop-Location
        Write-Host "done! all endpoints deployed." -ForegroundColor Green
    }
    
    "run-free" {
        Write-Host "running free tier test..." -ForegroundColor Blue
        Push-Location nvyra-x-free
        modal run inference.py
        Pop-Location
    }
    
    "run-pro" {
        Write-Host "running pro tier test..." -ForegroundColor Blue
        Push-Location nvyra-x-pro
        modal run inference.py
        Pop-Location
    }
    
    "build-rust" {
        Write-Host "building rust client (release mode with lto)..." -ForegroundColor Yellow
        Push-Location nvyra-x-rust
        cargo build --release
        Pop-Location
        Write-Host "binary location: nvyra-x-rust/target/release/nvyra-x-client.exe" -ForegroundColor Green
    }
    
    "benchmark" {
        if (-not $InputFile) {
            Write-Host "usage: .\deploy.ps1 benchmark <input.csv>" -ForegroundColor Red
            exit 1
        }
        
        Write-Host "running rust client benchmark..." -ForegroundColor Magenta
        Push-Location nvyra-x-rust
        $proUrl = [System.Environment]::GetEnvironmentVariable("NVYRA_PRO_URL")
        $freeUrl = [System.Environment]::GetEnvironmentVariable("NVYRA_FREE_URL")
        
        cargo run --release -- `
            --pro-url $proUrl `
            --free-url $freeUrl `
            --input $InputFile `
            --output "benchmark_results.jsonl" `
            --benchmark `
            --max-connections 500 `
            --batch-size 128 `
            --max-batch-size 512
        Pop-Location
    }
    
    default {
        Write-Host "usage: .\deploy.ps1 <command>" -ForegroundColor White
        Write-Host ""
        Write-Host "commands:" -ForegroundColor White
        Write-Host "  deploy-free   deploy free tier to modal" -ForegroundColor Gray
        Write-Host "  deploy-pro    deploy pro tier to modal" -ForegroundColor Gray
        Write-Host "  deploy-all    deploy all tiers to modal" -ForegroundColor Gray
        Write-Host "  run-free      test free tier locally" -ForegroundColor Gray
        Write-Host "  run-pro       test pro tier locally" -ForegroundColor Gray
        Write-Host "  build-rust    build rust client (release)" -ForegroundColor Gray
        Write-Host "  benchmark     run throughput benchmark" -ForegroundColor Gray
        Write-Host ""
    }
}

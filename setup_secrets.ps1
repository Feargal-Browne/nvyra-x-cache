$ErrorActionPreference = "Stop"

Write-Host "setting up modal secrets from .env..." -ForegroundColor Cyan

# 1. Parse .env file
$envData = @{}
if (Test-Path ".env") {
    Get-Content ".env" | ForEach-Object {
        if ($_ -match "^([^#][^=]+)=(.*)$") {
            $key = $matches[1].Trim()
            $value = $matches[2].Trim()
            $envData[$key] = $value
        }
    }
} else {
    Write-Error "no .env file found!"
}

# 2. Create secrets
if ($envData.ContainsKey("hf_token")) {
    Write-Host "creating huggingface-secret..." -ForegroundColor Yellow
    modal secret create huggingface-secret HF_TOKEN=$($envData["hf_token"]) --force
}

if ($envData.ContainsKey("turso_url") -and $envData.ContainsKey("turso_auth_token")) {
    Write-Host "creating turso-api-new..." -ForegroundColor Yellow
    modal secret create turso-api-new TURSO_URL=$($envData["turso_url"]) TURSO_AUTH_TOKEN=$($envData["turso_auth_token"]) --force
}

Write-Host "secrets setup complete!" -ForegroundColor Green

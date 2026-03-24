Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $projectRoot

function Get-EnvValue {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Key
    )
    if (-not (Test-Path -LiteralPath ".env")) {
        return $null
    }
    $line = Get-Content ".env" | Where-Object {
        $_ -match "^\s*$Key\s*="
    } | Select-Object -First 1
    if (-not $line) {
        return $null
    }
    $value = $line.Split("=", 2)[1].Trim()
    if ($value.StartsWith('"') -and $value.EndsWith('"')) {
        return $value.Trim('"')
    }
    return $value
}

function Test-Http200 {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Url
    )
    try {
        $resp = Invoke-WebRequest -UseBasicParsing -Uri $Url -TimeoutSec 8
        return ($resp.StatusCode -eq 200)
    }
    catch {
        return $false
    }
}

if (-not (Test-Path -LiteralPath ".env")) {
    Write-Host "[ERROR] .env not found. Create it from config/settings.env first." -ForegroundColor Red
    exit 1
}

$pythonExe = if (Test-Path -LiteralPath ".venv\Scripts\python.exe") {
    ".venv\Scripts\python.exe"
}
else {
    "python"
}

Write-Host "Starting Docker services (RSSHub, Redis)..." -ForegroundColor Cyan
docker compose up -d

$rsshubUrl = Get-EnvValue -Key "RSSHUB_URL"
if (-not $rsshubUrl) { $rsshubUrl = "http://localhost:1200" }

$lmStudioUrl = Get-EnvValue -Key "LM_STUDIO_URL"
if (-not $lmStudioUrl) { $lmStudioUrl = "http://localhost:1234/v1" }

$rsshubHealth = "$($rsshubUrl.TrimEnd('/'))/healthz"
$lmStudioModels = "$($lmStudioUrl.TrimEnd('/'))/models"

if (-not (Test-Http200 -Url $rsshubHealth)) {
    Write-Host "[ERROR] RSSHub is not ready at $rsshubHealth" -ForegroundColor Red
    exit 1
}

if (-not (Test-Http200 -Url $lmStudioModels)) {
    Write-Host "[ERROR] LM Studio API is not ready at $lmStudioModels" -ForegroundColor Red
    Write-Host "Open LM Studio, load a model, and click 'Start Server'." -ForegroundColor Yellow
    exit 1
}

Write-Host "All checks passed. Starting pipeline..." -ForegroundColor Green
Write-Host "Press Ctrl+C to stop." -ForegroundColor DarkGray

& $pythonExe "src/main.py"

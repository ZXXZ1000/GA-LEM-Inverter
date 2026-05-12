<# 
GA-LEM-Inverter Windows bootstrap and diagnostics.

Use this from Windows PowerShell or PowerShell 7:

  powershell -ExecutionPolicy Bypass -File .\setup_environment.ps1

The script does not require Git Bash, bash, or system pip. It installs or reuses
Miniconda, creates a project-local .conda environment, installs pinned packages,
registers a Jupyter kernel, and runs a Fastscape smoke test.
#>

[CmdletBinding()]
param(
    [switch]$KeepExisting,
    [switch]$Recreate,
    [switch]$NoJupyter,
    [switch]$DryRun,
    [switch]$DiagnoseOnly,
    [switch]$InstallGit,
    [string]$CondaRoot = "",
    [string]$CondaSolver = "libmamba",
    [string]$CondaRepodataFn = "current_repodata.json",
    [string]$CondaRemoteConnectTimeoutSecs = "30",
    [string]$CondaRemoteReadTimeoutSecs = "180",
    [string]$ProjectDir = ""
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$PythonVersion = "3.11"
$EnvDirName = ".conda"
$KernelName = "ga-lem-inverter"
$KernelDisplayName = "GA-LEM-Inverter (Python $PythonVersion)"

if ([string]::IsNullOrWhiteSpace($CondaRoot)) {
    $CondaRoot = Join-Path $HOME "miniconda3"
}

function Write-Info($Message) { Write-Host "[INFO] $Message" -ForegroundColor Blue }
function Write-Success($Message) { Write-Host "[SUCCESS] $Message" -ForegroundColor Green }
function Write-WarningLine($Message) { Write-Host "[WARNING] $Message" -ForegroundColor Yellow }
function Write-ErrorLine($Message) { Write-Host "[ERROR] $Message" -ForegroundColor Red }

function Start-SetupLog {
    $script:LogFile = Join-Path $ProjectRoot "setup_environment.log"
    if (Test-Path $script:LogFile) {
        Remove-Item -Force $script:LogFile
    }
    Start-Transcript -Path $script:LogFile -Force | Out-Null
    Write-Info "Log file: $script:LogFile"
}

function Stop-SetupLog {
    try {
        Stop-Transcript | Out-Null
    } catch {
    }
}

function Get-CommandPath($Name) {
    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    return $null
}

function Test-CommandPresent($Name, $Required = $false) {
    $path = Get-CommandPath $Name
    if ($path) {
        Write-Success "$Name found: $path"
        return $true
    }
    if ($Required) {
        Write-ErrorLine "$Name not found"
    } else {
        Write-WarningLine "$Name not found"
    }
    return $false
}

function Get-CondaPath {
    $candidates = @(
        (Join-Path $CondaRoot "Scripts\conda.exe"),
        (Join-Path $CondaRoot "bin\conda")
    )
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) { return $candidate }
    }

    $cmd = Get-CommandPath "conda"
    if ($cmd) { return $cmd }

    $bat = Join-Path $CondaRoot "condabin\conda.bat"
    if (Test-Path $bat) { return $bat }

    return $null
}

function Install-GitIfRequested {
    if (-not $InstallGit) { return }
    if (Get-CommandPath "git") { return }

    $winget = Get-CommandPath "winget"
    if (-not $winget) {
        Write-WarningLine "Git is missing and winget is not available. Download Git manually from https://git-scm.com/download/win if needed."
        return
    }

    Write-Info "Installing Git for Windows through winget"
    & winget install --id Git.Git -e --source winget --accept-package-agreements --accept-source-agreements
}

function Install-MinicondaIfNeeded {
    $script:CondaBin = Get-CondaPath
    if ($script:CondaBin) {
        Write-Info "Using conda manager: $script:CondaBin"
        Write-Info "Target packages are installed into: $EnvPath"
        return
    }

    $installer = "Miniconda3-latest-Windows-x86_64.exe"
    $url = "https://repo.anaconda.com/miniconda/$installer"
    $installerPath = Join-Path $ProjectRoot $installer

    Write-Info "Miniconda not found. Downloading $installer"
    Invoke-WebRequest -Uri $url -OutFile $installerPath

    Write-Info "Installing Miniconda to $CondaRoot"
    New-Item -ItemType Directory -Force -Path (Split-Path $CondaRoot -Parent) | Out-Null
    $args = @(
        "/InstallationType=JustMe",
        "/RegisterPython=0",
        "/AddToPath=0",
        "/S",
        "/D=$CondaRoot"
    )
    $process = Start-Process -FilePath $installerPath -ArgumentList $args -Wait -PassThru
    Remove-Item -Force $installerPath

    if ($process.ExitCode -ne 0) {
        throw "Miniconda installer failed with exit code $($process.ExitCode)"
    }

    $script:CondaBin = Get-CondaPath
    if (-not $script:CondaBin) {
        throw "Conda installation finished but conda executable was not found."
    }
    Write-Success "Miniconda installed: $script:CondaBin"
}

function Invoke-Conda {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    & $script:CondaBin @Arguments
    if ($LASTEXITCODE -eq 0) {
        return
    }

    $firstExitCode = $LASTEXITCODE
    if (-not [string]::IsNullOrWhiteSpace($CondaSolver) -and ($Arguments -contains "--solver")) {
        Write-WarningLine "conda failed with solver '$CondaSolver'; retrying with conda's default solver."
        $filteredArgs = New-Object System.Collections.Generic.List[string]
        for ($i = 0; $i -lt $Arguments.Count; $i++) {
            if ($Arguments[$i] -eq "--solver") {
                $i++
                continue
            }
            $filteredArgs.Add($Arguments[$i])
        }
        & $script:CondaBin @filteredArgs
        if ($LASTEXITCODE -eq 0) {
            return
        }
    }

    throw "conda failed with exit code $firstExitCode"
}

function Get-CondaHelpText {
    param([string]$Command)
    $output = & $script:CondaBin $Command --help 2>$null
    if ($LASTEXITCODE -ne 0) { return "" }
    return ($output -join "`n")
}

function Configure-CondaArgs {
    $env:CONDA_REMOTE_CONNECT_TIMEOUT_SECS = $CondaRemoteConnectTimeoutSecs
    $env:CONDA_REMOTE_READ_TIMEOUT_SECS = $CondaRemoteReadTimeoutSecs

    $createHelp = Get-CondaHelpText "create"
    $installHelp = Get-CondaHelpText "install"

    $script:CondaCreateArgs = @("--override-channels", "-c", "conda-forge", "--strict-channel-priority")
    $script:CondaInstallArgs = @("--override-channels", "-c", "conda-forge", "--strict-channel-priority")

    if ($createHelp -match "--repodata-fn") {
        $script:CondaCreateArgs += @("--repodata-fn", $CondaRepodataFn)
    }
    if ($installHelp -match "--repodata-fn") {
        $script:CondaInstallArgs += @("--repodata-fn", $CondaRepodataFn)
    }

    if (-not [string]::IsNullOrWhiteSpace($CondaSolver) -and $createHelp -match "--solver") {
        $script:CondaCreateArgs += @("--solver", $CondaSolver)
    }
    if (-not [string]::IsNullOrWhiteSpace($CondaSolver) -and $installHelp -match "--solver") {
        $script:CondaInstallArgs += @("--solver", $CondaSolver)
    }

    if ($installHelp -match "--satisfied-skip-solve") {
        $script:CondaInstallArgs += "--satisfied-skip-solve"
    }

    Write-Info "Conda create options: $($script:CondaCreateArgs -join ' ')"
    Write-Info "Conda install options: $($script:CondaInstallArgs -join ' ')"
    Write-Info "Conda network timeouts: connect=${CondaRemoteConnectTimeoutSecs}s read=${CondaRemoteReadTimeoutSecs}s"
}

function Resolve-ProjectRoot {
    if (-not [string]::IsNullOrWhiteSpace($ProjectDir)) {
        $candidate = $ProjectDir
    } else {
        $candidate = $PSScriptRoot
        if ([string]::IsNullOrWhiteSpace($candidate)) {
            $candidate = Split-Path -Parent $PSCommandPath
        }
        if ([string]::IsNullOrWhiteSpace($candidate)) {
            $candidate = (Get-Location).Path
        }
    }

    if (-not (Test-Path $candidate -PathType Container)) {
        throw "Project directory does not exist: $candidate"
    }

    return (Resolve-Path $candidate).Path
}

function Assert-ProjectRoot {
    $missing = @()
    foreach ($file in @("main.py", "config.ini", "setup_environment.ps1")) {
        if (-not (Test-Path (Join-Path $ProjectRoot $file) -PathType Leaf)) {
            $missing += $file
        }
    }
    if ($missing.Count -gt 0) {
        throw "Project root check failed: $ProjectRoot. Missing required project files: $($missing -join ', '). Pass -ProjectDir C:\path\to\GA-LEM-Inverter if this script was launched from a wrapper."
    }
}

function Assert-EnvPrefix {
    if (-not (Test-Path $EnvPython -PathType Leaf)) {
        throw "Expected environment Python was not found: $EnvPython. The environment must be created under the project root: $EnvPath"
    }

    $checkCode = @'
import os
import sys

expected = os.path.normcase(os.path.realpath(os.path.abspath(sys.argv[1])))
actual = os.path.normcase(os.path.realpath(os.path.abspath(sys.prefix)))

if actual != expected:
    print(f"Environment prefix mismatch: expected {expected}, got {actual}", file=sys.stderr)
    sys.exit(1)

print(f"Environment prefix verified: {sys.prefix}")
'@
    & $EnvPython -c $checkCode $EnvPath
    if ($LASTEXITCODE -ne 0) {
        throw "Environment prefix validation failed"
    }
}

function Invoke-EnvPython {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    & $EnvPython @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "python failed with exit code $LASTEXITCODE"
    }
}

function Run-Diagnostics {
    Write-Info "Running host diagnostics"
    Write-Info "Current dir: $((Get-Location).Path)"
    Write-Info "Project dir: $ProjectRoot"
    Write-Info "PowerShell: $($PSVersionTable.PSVersion)"
    Write-Info "OS: $([System.Environment]::OSVersion.VersionString)"
    Write-Info "Architecture: $([System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture)"
    Test-CommandPresent "git" | Out-Null
    Test-CommandPresent "bash" | Out-Null
    Test-CommandPresent "pip" | Out-Null
    Test-CommandPresent "python" | Out-Null
    Test-CommandPresent "winget" | Out-Null
    $conda = Get-CondaPath
    if ($conda) {
        Write-Success "conda found: $conda"
    } else {
        Write-WarningLine "conda not found; this script can install Miniconda automatically."
    }
}

function Diagnose-CondaManager {
    Write-Info "Conda manager diagnostics"
    $conda = Get-CondaPath
    if ($conda) {
        Write-Success "Preferred conda manager: $conda"
        & $conda --version
        Write-Info "Target environment prefix: $EnvPath"
    } else {
        Write-WarningLine "No conda manager found. Setup will install Miniconda to: $CondaRoot"
        Write-Info "Target environment prefix after install: $EnvPath"
    }
}

$CondaPackages = @(
    "python=$PythonVersion",
    "numpy=2.3.5",
    "scipy=1.17.1",
    "matplotlib=3.10.9",
    "pandas=3.0.2",
    "scikit-image=0.26.0",
    "scikit-learn=1.8.0",
    "xarray=2026.4.0",
    "xarray-simlab=0.5.0",
    "fastscape=0.1.0",
    "zarr=2.18.7",
    "numcodecs=0.15.1",
    "numba=0.65.1",
    "llvmlite=0.47.0",
    "rasterio=1.4.4",
    "geopandas=1.1.3",
    "shapely=2.1.2",
    "affine=2.4.0",
    "pyproj=3.7.2",
    "cartopy=0.25.0",
    "libpysal=4.14.1",
    "esda=2.9.0",
    "seaborn=0.13.2",
    "tqdm=4.67.3",
    "ipywidgets=8.1.8",
    "notebook=7.5.6",
    "ipykernel=7.2.0",
    "psutil=7.2.2",
    "joblib=1.5.3",
    "typeguard=4.5.1",
    "pyyaml=6.0.3",
    "dask=2026.3.0",
    "plotly=6.6.0",
    "pytest=9.0.3",
    "black=26.3.1",
    "flake8=7.3.0",
    "mypy=1.20.2"
)

$PipPackages = @(
    "torch==2.11.0",
    "torchvision==0.26.0",
    "lpips==0.1.4",
    "opencv-python==4.13.0.92",
    "pykrige==1.7.3",
    "scikit-opt==0.6.6"
)

function Print-Plan {
    Write-Info "Project dir: $ProjectRoot"
    Write-Info "Conda root: $CondaRoot"
    Write-Info "Conda solver preference: $CondaSolver"
    Write-Info "Conda repodata preference: $CondaRepodataFn"
    Write-Info "Conda network timeouts: connect=${CondaRemoteConnectTimeoutSecs}s read=${CondaRemoteReadTimeoutSecs}s"
    Write-Info "Environment: $EnvPath"
    Write-Info "Recreate env: $ShouldRecreate"
    Write-Info "Jupyter kernel: $(-not $NoJupyter)"
    Write-Info "Conda packages:"
    $CondaPackages | ForEach-Object { Write-Host "  $_" }
    Write-Info "Pip packages:"
    $PipPackages | ForEach-Object { Write-Host "  $_" }
}

function Diagnose-ExistingEnvironment {
    Write-Info "Project-local environment diagnostics"
    if (-not (Test-Path $EnvPath -PathType Container)) {
        Write-WarningLine "Environment directory does not exist yet: $EnvPath"
        return
    }

    Write-Success "Environment directory exists: $EnvPath"
    if (-not (Test-Path $EnvPython -PathType Leaf)) {
        Write-ErrorLine "Environment Python is missing: $EnvPython"
        return
    }

    $diagnoseCode = @'
import importlib.metadata as md
import os
import sys

expected = os.path.normcase(os.path.realpath(os.path.abspath(sys.argv[1])))
actual = os.path.normcase(os.path.realpath(os.path.abspath(sys.prefix)))

print(f"python_executable={sys.executable}")
print(f"sys_prefix={sys.prefix}")
print(f"expected_prefix={sys.argv[1]}")
print(f"prefix_match={actual == expected}")

checks = {
    "numpy": "2.3.5",
    "zarr": "2.18.7",
    "xarray-simlab": "0.5.0",
    "fastscape": "0.1.0",
    "notebook": "7.5.6",
    "ipykernel": "7.2.0",
}
ok = actual == expected
for package, expected_version in checks.items():
    try:
        actual_version = md.version(package)
    except md.PackageNotFoundError:
        print(f"{package}=MISSING expected={expected_version}")
        ok = False
        continue
    print(f"{package}={actual_version} expected={expected_version}")
    if actual_version != expected_version:
        ok = False

try:
    import numpy as np
    import zarr
    print(f"numpy_has_in1d={hasattr(np, 'in1d')}")
    print(f"zarr_has_MemoryStore={hasattr(zarr, 'MemoryStore')}")
    ok = ok and hasattr(np, "in1d") and hasattr(zarr, "MemoryStore")
except Exception as exc:
    print(f"compatibility_import_error={exc}")
    ok = False

sys.exit(0 if ok else 1)
'@
    & $EnvPython -c $diagnoseCode $EnvPath
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Existing .conda environment matches critical checks"
    } else {
        Write-WarningLine "Existing .conda environment is missing or mismatched; setup will recreate/update it unless -DiagnoseOnly was used."
    }
}

function Create-OrUpdateEnvironment {
    if ((Test-Path $EnvPath) -and $ShouldRecreate) {
        if ($env:CONDA_PREFIX -and ([System.IO.Path]::GetFullPath($env:CONDA_PREFIX) -eq [System.IO.Path]::GetFullPath($EnvPath))) {
            throw "The target environment is currently active. Deactivate it before recreating."
        }
        Write-WarningLine "Removing existing environment: $EnvPath"
        Remove-Item -Recurse -Force $EnvPath
    }

    if (-not (Test-Path $EnvPath)) {
        Write-Info "Creating conda environment at $EnvPath"
        $args = @("create", "-p", $EnvPath, "-y") + $script:CondaCreateArgs + $CondaPackages
        Invoke-Conda @args
    } else {
        Write-Info "Installing/updating conda packages in $EnvPath"
        $args = @("install", "-p", $EnvPath, "-y") + $script:CondaInstallArgs + $CondaPackages
        Invoke-Conda @args
    }

    Assert-EnvPrefix

    Write-Info "Upgrading pip inside project environment"
    Invoke-EnvPython "-m" "pip" "install" "--upgrade" "pip>=24,<26"

    Write-Info "Installing pinned pip packages"
    $pipArgs = @("-m", "pip", "install") + $PipPackages
    Invoke-EnvPython @pipArgs

    Assert-EnvPrefix
}

function Register-JupyterKernel {
    if ($NoJupyter) {
        Write-Info "Skipping Jupyter kernel registration"
        return
    }

    Write-Info "Registering Jupyter kernel"
    Invoke-EnvPython "-m" "ipykernel" "install" "--user" "--name=$KernelName" "--display-name=$KernelDisplayName"
}

function Verify-Environment {
    Write-Info "Verifying imports and Fastscape runtime compatibility"
    Assert-EnvPrefix
    $verifyCode = @'
import importlib.metadata as md
import numpy as np
import scipy
import matplotlib
import skimage
import sklearn
import rasterio
import geopandas
import shapely
import pyproj
import pandas
import xsimlab
import fastscape
import zarr
import torch
import torchvision
import lpips
import cv2
import tqdm
import dask
import plotly
import pykrige
import cartopy
import libpysal
import esda
import seaborn

assert hasattr(zarr, "MemoryStore"), "zarr<3 is required for xarray-simlab"
assert hasattr(np, "in1d"), "numpy<2.4 is required for xarray-simlab"

from ga_lem_inverter.pipeline.forward_model import run_fastscape_model

shape = (10, 10)
k_sp = np.full(shape, 6.92e-6)
uplift = np.full(shape, 5.0)
elevation = run_fastscape_model(
    k_sp=k_sp,
    uplift=uplift,
    k_diff=19.2,
    x_size=shape[1],
    y_size=shape[0],
    spacing=900,
    time_total=1e4,
)
assert elevation.shape == shape
assert np.isfinite(elevation).all()

packages = [
    "numpy", "scipy", "matplotlib", "scikit-image", "scikit-learn",
    "rasterio", "geopandas", "shapely", "pyproj", "pandas",
    "xarray-simlab", "fastscape", "zarr", "torch", "torchvision",
    "lpips", "opencv-python", "PyKrige",
]
for package in packages:
    print(f"{package}=={md.version(package)}")
try:
    import sko
    print(f"scikit-opt=={md.version('scikit-opt')}")
except Exception as exc:
    print(f"scikit-opt=={md.version('scikit-opt')} (optional import warning: {exc})")
print("Fastscape smoke test passed")
'@
    Push-Location $ProjectRoot
    try {
        $verifyCode | & $EnvPython -
        if ($LASTEXITCODE -ne 0) {
            throw "verification failed with exit code $LASTEXITCODE"
        }
    } finally {
        Pop-Location
    }
}

$ProjectRoot = Resolve-ProjectRoot
Assert-ProjectRoot
$EnvPath = Join-Path $ProjectRoot $EnvDirName
$EnvPython = Join-Path $EnvPath "python.exe"
$ShouldRecreate = -not $KeepExisting
if ($Recreate) { $ShouldRecreate = $true }

try {
    Start-SetupLog
    Run-Diagnostics
    Diagnose-CondaManager
    Install-GitIfRequested
    Print-Plan
    Diagnose-ExistingEnvironment

    if ($DiagnoseOnly) {
        Write-Success "Diagnosis completed; no environment changes made."
        Write-Info "Log file: $script:LogFile"
        exit 0
    }

    if ($DryRun) {
        Write-Success "Dry run completed; no changes made."
        Write-Info "Log file: $script:LogFile"
        exit 0
    }

    Install-MinicondaIfNeeded
    Configure-CondaArgs
    Create-OrUpdateEnvironment
    Register-JupyterKernel
    Verify-Environment

    Write-Success "Environment setup completed."
    Write-Info "Use this interpreter: $EnvPython"
    Write-Info "Or activate with: conda activate $EnvPath"
    Write-Info "Log file: $script:LogFile"
} finally {
    Stop-SetupLog
}

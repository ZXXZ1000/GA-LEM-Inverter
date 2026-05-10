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
    [string]$CondaRoot = ""
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
    $cmd = Get-CommandPath "conda"
    if ($cmd) { return $cmd }

    $candidates = @(
        (Join-Path $CondaRoot "Scripts\conda.exe"),
        (Join-Path $CondaRoot "condabin\conda.bat"),
        (Join-Path $CondaRoot "bin\conda")
    )
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) { return $candidate }
    }
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
        Write-Info "Using conda: $script:CondaBin"
        return
    }

    $installer = "Miniconda3-latest-Windows-x86_64.exe"
    $url = "https://repo.anaconda.com/miniconda/$installer"
    $installerPath = Join-Path $ScriptDir $installer

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
    if ($script:CondaBin -like "*.bat") {
        & cmd.exe /d /c "`"$script:CondaBin`" $($Arguments -join ' ')"
    } else {
        & $script:CondaBin @Arguments
    }
    if ($LASTEXITCODE -ne 0) {
        throw "conda failed with exit code $LASTEXITCODE"
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
    Write-Info "Project dir: $ScriptDir"
    Write-Info "Conda root: $CondaRoot"
    Write-Info "Environment: $EnvPath"
    Write-Info "Recreate env: $ShouldRecreate"
    Write-Info "Jupyter kernel: $(-not $NoJupyter)"
    Write-Info "Conda packages:"
    $CondaPackages | ForEach-Object { Write-Host "  $_" }
    Write-Info "Pip packages:"
    $PipPackages | ForEach-Object { Write-Host "  $_" }
}

function Create-OrUpdateEnvironment {
    if ((Test-Path $EnvPath) -and $ShouldRecreate) {
        if ($env:CONDA_PREFIX -and ([System.IO.Path]::GetFullPath($env:CONDA_PREFIX) -eq [System.IO.Path]::GetFullPath($EnvPath))) {
            throw "The target environment is currently active. Deactivate it before recreating."
        }
        Write-WarningLine "Removing existing environment: $EnvPath"
        Remove-Item -Recurse -Force $EnvPath
    }

    $commonArgs = @("-p", $EnvPath, "-y", "--override-channels", "-c", "conda-forge", "--strict-channel-priority")
    if (-not (Test-Path $EnvPath)) {
        Write-Info "Creating conda environment at $EnvPath"
        $args = @("create") + $commonArgs + $CondaPackages
        Invoke-Conda @args
    } else {
        Write-Info "Installing/updating conda packages in $EnvPath"
        $args = @("install") + $commonArgs + $CondaPackages
        Invoke-Conda @args
    }

    Write-Info "Upgrading pip inside project environment"
    Invoke-EnvPython "-m" "pip" "install" "--upgrade" "pip>=24,<26"

    Write-Info "Installing pinned pip packages"
    $pipArgs = @("-m", "pip", "install") + $PipPackages
    Invoke-EnvPython @pipArgs
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

from model_runner import run_fastscape_model

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
    Push-Location $ScriptDir
    try {
        $verifyCode | & $EnvPython -
        if ($LASTEXITCODE -ne 0) {
            throw "verification failed with exit code $LASTEXITCODE"
        }
    } finally {
        Pop-Location
    }
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if ([string]::IsNullOrWhiteSpace($ScriptDir)) {
    $ScriptDir = (Get-Location).Path
}
$EnvPath = Join-Path $ScriptDir $EnvDirName
$EnvPython = Join-Path $EnvPath "python.exe"
$ShouldRecreate = -not $KeepExisting
if ($Recreate) { $ShouldRecreate = $true }

Run-Diagnostics
Install-GitIfRequested
Print-Plan

if ($DiagnoseOnly) {
    Write-Success "Diagnosis completed; no environment changes made."
    exit 0
}

if ($DryRun) {
    Write-Success "Dry run completed; no changes made."
    exit 0
}

Install-MinicondaIfNeeded
Create-OrUpdateEnvironment
Register-JupyterKernel
Verify-Environment

Write-Success "Environment setup completed."
Write-Info "Use this interpreter: $EnvPython"

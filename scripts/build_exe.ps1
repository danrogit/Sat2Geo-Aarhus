param(
  [string]$Name = "Sat2Geo-Aarhus",
  [ValidateSet("onedir", "onefile")]
  [string]$PackageMode = "onedir"
)

$ErrorActionPreference = "Stop"

python -m pip install -r requirements-gui.txt

$modeArg = if ($PackageMode -eq "onefile") { "--onefile" } else { "--onedir" }

python -m PyInstaller `
  --noconfirm `
  --name $Name `
  $modeArg `
  --windowed `
  --add-data "data;data" `
  --add-data "sat2geo;sat2geo" `
  --collect-all transformers `
  --collect-all torch `
  --collect-all faiss `
  sat2geo_gui.py

if ($PackageMode -eq "onefile") {
  Write-Host "Built dist\$Name.exe"
} else {
  Write-Host "Built dist\$Name\$Name.exe"
}

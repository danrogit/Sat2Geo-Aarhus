# Building the Windows EXE

The GUI is a small Tkinter wrapper around the same Sat2Geo locator used by the CLI.

## Build

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
.\scripts\build_exe.ps1
```

The clickable app will be written to:

```text
dist\Sat2Geo-Aarhus\Sat2Geo-Aarhus.exe
```

## Notes

The default build is a folder-style Windows app because PyTorch, Transformers, Faiss, and the Aarhus demo index are large. Users can still double-click `Sat2Geo-Aarhus.exe`; the surrounding files are its runtime.

To force a single huge `.exe`, run:

```powershell
.\scripts\build_exe.ps1 -PackageMode onefile
```

The first run may download the DINOv2 model into the user's Hugging Face cache unless it is already cached.

For a smaller app, ship the CLI and ask users to install Python dependencies. For a friendlier app, keep the EXE.

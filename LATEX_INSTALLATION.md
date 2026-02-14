# LaTeX Installation Guide

## Problem
You're getting the error: **"Recipe terminated with fatal error: spawn latexmk ENOENT"**

This means LaTeX is not installed on your system. The LaTeX Workshop extension cannot find `latexmk` (or any LaTeX tools) in your PATH.

## Solution: Install a LaTeX Distribution

### Option 1: MiKTeX (Recommended for Windows) ⭐

**Best for:** Most users, easy installation, automatic package management

1. **Download MiKTeX:**
   - Visit: https://miktex.org/download
   - Download the **Basic MiKTeX Installer** (recommended) or **Complete Installer**

2. **Install MiKTeX:**
   - Run the installer
   - **IMPORTANT:** During installation, check the option **"Add MiKTeX to PATH"**
   - Choose "Install missing packages automatically" when prompted
   - Complete the installation

3. **Restart VS Code/Cursor:**
   - Close and reopen VS Code/Cursor completely
   - This ensures the new PATH is loaded

4. **Verify Installation:**
   ```powershell
   latexmk --version
   pdflatex --version
   ```
   If these commands work, LaTeX is installed correctly!

### Option 2: TeX Live

**Best for:** Advanced users, complete distribution

1. **Download TeX Live:**
   - Visit: https://www.tug.org/texlive/windows.html
   - Download the installer

2. **Install TeX Live:**
   - Run `install-tl-windows.exe`
   - Follow the installation wizard
   - **IMPORTANT:** Ensure TeX Live is added to PATH during installation
   - Installation may take 30-60 minutes (it's a large distribution)

3. **Restart VS Code/Cursor**

4. **Verify Installation:**
   ```powershell
   latexmk --version
   ```

### Option 3: Use Overleaf (Online) 🌐

**Best for:** Quick start, no installation needed

As mentioned in your thesis document (line 136), **Overleaf is the recommended option**:

1. Visit: https://www.overleaf.com/
2. Create a free account
3. Upload your thesis files
4. Compile online - no setup required!

**Advantages:**
- No installation needed
- All packages pre-installed
- Collaborative editing
- Version control built-in
- Works on any device

## After Installation

Once LaTeX is installed:

1. **Restart VS Code/Cursor** (important!)

2. **Re-enable auto-build** (if you want):
   - Open `.vscode/settings.json`
   - Change `"latex-workshop.latex.autoBuild.run": "never"` to `"onFileChange"` or `"onSave"`

3. **Compile your document:**
   - Press `Ctrl+Alt+B` to build
   - Or use Command Palette: `LaTeX Workshop: Build LaTeX project`

## Current Configuration

Your VS Code settings are already configured in `.vscode/settings.json`:
- Root file: `thesis/main.tex`
- Build recipes: latexmk and pdflatex+bibtex
- PDF viewer: Internal tab viewer

## Troubleshooting

### If `latexmk` still not found after installation:

1. **Check PATH:**
   ```powershell
   $env:Path -split ';' | Select-String -Pattern "miktex|texlive"
   ```

2. **Manually add to PATH** (if needed):
   - MiKTeX: Usually `C:\Users\<YourName>\AppData\Local\Programs\MiKTeX\miktex\bin\x64`
   - TeX Live: Usually `C:\texlive\<year>\bin\win32`

3. **Restart VS Code/Cursor** after PATH changes

### If you see "kpsewhich returned with non-zero code 1":
- This means LaTeX is not properly installed or not in PATH
- Follow the installation steps above

## Need Help?

- MiKTeX Documentation: https://miktex.org/kb/
- TeX Live Documentation: https://www.tug.org/texlive/doc.html
- LaTeX Workshop Extension: https://github.com/James-Yu/LaTeX-Workshop


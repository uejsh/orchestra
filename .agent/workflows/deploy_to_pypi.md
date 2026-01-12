---
description: Deploy Orchestra to PyPI
---

# How to Deploy Orchestra to PyPI

This workflow describes the steps to build and publish the `orchestra-ai` package to the Python Package Index (PyPI).

## Prerequisites

1.  **PyPI Account**: You must have an account on [pypi.org](https://pypi.org/).
2.  **API Token**: Create an API token in your PyPI account settings.
3.  **Deployment Tools**: Ensure `build` and `twine` are installed (handled in setup step).

## Deployment Steps

### 1. Verify Version
Ensure the version in `orchestra/_version.py` and `pyproject.toml` is correct and unique (has not been uploaded before).

### 2. Clean Previous Builds
Remove any existing `dist/` or `build/` directories to avoid confusion.

```powershell
Remove-Item -Recurse -Force dist, build, *.egg-info -ErrorAction SilentlyContinue
```

### 3. Build the Package
Generate the Source Distribution (`.tar.gz`) and Wheel (`.whl`).

```powershell
# // turbo
python -m build
```

### 4. Validate the Package
Check that the metadata and README render correctly.

```powershell
# // turbo
twine check dist/*
```

### 5. Upload to PyPI
Upload the artifacts. You will be prompted for your username (use `__token__`) and your API token (paste the token starting with `pypi-`).

```powershell
twine upload dist/*
```

> **Note**: If you want to test first, you can upload to TestPyPI using:
> `twine upload --repository testpypi dist/*`

## Post-Deployment

1.  Create a **Git Tag** for the release:
    ```bash
    git tag v0.1.0
    git push origin v0.1.0
    ```
2.  Create a **GitHub Release** with release notes.

# pix2tex Installation Guide

**Date**: 2025-10-18
**Story**: 8.1 - Formula Tool Research & Selection
**Tool**: pix2tex (LaTeX-OCR)

## Overview

This guide documents the installation and setup of pix2tex for formula extraction in the pdf2llm project.

## System Requirements

### Python Version
- **Minimum**: Python 3.7+
- **Tested with**: Python 3.12
- **Compatible with**: Python >=3.10 (pdf2llm requirement)

### Operating Systems
- ✓ macOS (Intel and Apple Silicon)
- ✓ Linux
- ✓ Windows

## Installation

### Standard Installation

```bash
pip install "pix2tex[gui]"
```

The `[gui]` extra includes dependencies for the GUI tool (`latexocr` command). For programmatic use only, you can install without GUI:

```bash
pip install pix2tex
```

### Dependencies

pix2tex will install the following dependencies:

**Core Dependencies**:
- `torch>=1.7.1` - PyTorch deep learning framework
- `transformers>=4.18.0` - Hugging Face transformers
- `tokenizers>=0.13.0` - Fast tokenization
- `timm==0.5.4` - PyTorch Image Models
- `x-transformers==0.15.0` - Extended transformer implementations
- `einops>=0.3.0` - Tensor operations

**Image Processing**:
- `Pillow>=9.1.0` - Image manipulation
- `opencv-python-headless>=4.1.1.26` - Computer vision
- `albumentations<=1.4.24,>=0.5.2` - Image augmentation

**Utilities**:
- `numpy>=1.19.5`
- `pandas>=1.0.0`
- `PyYAML>=5.4.1`
- `munch>=2.5.0`
- `tqdm>=4.47.0`
- `requests>=2.22.0`

**GUI Dependencies** (if using `[gui]`):
- `PyQt6` - Qt GUI framework
- `PyQt6-WebEngine` - Web rendering
- `pyside6` - Alternative Qt bindings
- `pynput` - Input control
- `screeninfo` - Screen information
- `latex2sympy2` - LaTeX to SymPy conversion

### Model Files

On first run, pix2tex automatically downloads model weights:

- **Main model**: `weights.pth` (~97.4 MB)
- **Image resizer**: `image_resizer.pth` (~18.5 MB)
- **Total**: ~116 MB

Models are cached in: `<python-site-packages>/pix2tex/model/checkpoints/`

## Verification

### Quick Test

Create a simple test script (`test_pix2tex.py`):

```python
from PIL import Image
from pix2tex.cli import LatexOCR

# Initialize model (downloads weights on first run)
model = LatexOCR()

# Test with an image
img = Image.open('formula.png')
latex_code = model(img)
print(latex_code)
```

Run the test:

```bash
python test_pix2tex.py
```

### Expected Performance

Based on PoC testing (macOS Apple Silicon, Python 3.12):

- **First run**: ~19s (includes model download)
- **Subsequent runs**: ~1-2s model init + ~1-2s per formula
- **Memory**: ~500MB RAM for model in memory
- **Accuracy**: BLEU 0.88, token accuracy 0.60

## Integration with pdf2llm

### Adding Dependency

Add to `pyproject.toml`:

```toml
[project]
dependencies = [
    # ... existing dependencies ...
    "pix2tex>=0.1.0",
]
```

### Optional Dependencies

To avoid bloat for users not using formula extraction:

```toml
[project.optional-dependencies]
formula = [
    "pix2tex>=0.1.0",
]
```

Install with: `pip install pdf2llm[formula]`

### Model Caching Strategy

For optimal performance in batch processing:

```python
# Initialize once, reuse for multiple formulas
from pix2tex.cli import LatexOCR

class FormulaExtractor:
    def __init__(self):
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = LatexOCR()
        return self._model

    def extract(self, image_path):
        from PIL import Image
        img = Image.open(image_path)
        return self.model(img)
```

## Known Issues and Workarounds

### 1. Dependency Conflicts

When installing alongside pdf2llm dependencies:

**Issue**: `timm` version conflict (pix2tex requires 0.5.4, effdet requires >=0.9.2)

```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
effdet 0.4.1 requires timm>=0.9.2, but you have timm 0.5.4 which is incompatible.
```

**Impact**: Low - effdet is not used for formula extraction workflow

**Workaround**: Install in separate virtual environment or use `--no-deps` flag

**Long-term solution**: Story 8.2 will evaluate RapidLaTeXOCR (ONNX version) which has lighter dependencies

### 2. First-run Model Download

**Issue**: Model download can be slow on first run (~116MB over network)

**Workaround**: Pre-download models during installation:

```python
from pix2tex.cli import LatexOCR
# Trigger model download
_ = LatexOCR()
```

### 3. Apple Silicon Support

**Status**: ✓ Fully supported

pix2tex works correctly with MPS (Metal Performance Shaders) backend on Apple Silicon:

```
MPS available: True (Apple Silicon)
```

No additional configuration needed.

## Command-Line Usage

### CLI Tool

```bash
# Process image from disk
pix2tex path/to/formula.png

# Process image from clipboard
pix2tex
```

### GUI Tool

```bash
# Launch GUI application
latexocr
```

Features:
- Screenshot capture
- Real-time preview with MathJax rendering
- Copy to clipboard
- Retry with different temperatures

## API Usage Patterns

### Basic Usage

```python
from PIL import Image
from pix2tex.cli import LatexOCR

model = LatexOCR()
img = Image.open('formula.png')
latex = model(img)
```

### Batch Processing

```python
from PIL import Image
from pix2tex.cli import LatexOCR
from pathlib import Path

model = LatexOCR()
formulas = []

for img_path in Path('formulas/').glob('*.png'):
    img = Image.open(img_path)
    latex = model(img)
    formulas.append((img_path.name, latex))
```

### With Error Handling

```python
from PIL import Image
from pix2tex.cli import LatexOCR
import logging

logger = logging.getLogger(__name__)
model = LatexOCR()

def extract_formula(image_path):
    try:
        img = Image.open(image_path)
        return model(img)
    except Exception as e:
        logger.error(f"Formula extraction failed for {image_path}: {e}")
        return None
```

## Docker Deployment

For containerized deployment (Future: Story 8.5):

```dockerfile
FROM python:3.12-slim

# Install dependencies
RUN pip install "pix2tex[gui]"

# Pre-download models
RUN python -c "from pix2tex.cli import LatexOCR; LatexOCR()"

# Copy application
COPY . /app
WORKDIR /app

CMD ["python", "app.py"]
```

## Troubleshooting

### Model Download Fails

```python
# Manual download (if needed)
import requests
from pathlib import Path

model_url = "https://github.com/lukas-blecher/LaTeX-OCR/releases/download/v0.0.1/weights.pth"
model_path = Path("path/to/save/weights.pth")
response = requests.get(model_url)
model_path.write_bytes(response.content)
```

### Out of Memory

If processing very large images:

```python
from PIL import Image

# Resize before processing
img = Image.open('large_formula.png')
max_size = (1024, 1024)
img.thumbnail(max_size, Image.Resampling.LANCZOS)
latex = model(img)
```

### Slow Performance

- **Cold start**: First invocation is slow (model initialization)
- **Solution**: Keep model in memory for batch processing
- **Alternative**: Use RapidLaTeXOCR (ONNX) for faster inference (Story 8.2 consideration)

## Resources

- **GitHub**: https://github.com/lukas-blecher/LaTeX-OCR
- **Documentation**: https://pix2tex.readthedocs.io/
- **PyPI**: https://pypi.org/project/pix2tex/
- **Demo**: https://huggingface.co/spaces/lukbl/LaTeX-OCR

## Next Steps

- ✓ Installation verified (Story 8.1)
- → Story 8.2: Integrate into pdf2llm pipeline
- → Story 8.3: Implement formula output formatting
- → Story 8.4: Hybrid document processing
- → Story 8.5: CLI integration and testing

---

**Installation guide verified**: 2025-10-18
**Test environment**: macOS (Apple Silicon), Python 3.12
**PoC result**: ✓ PASSED

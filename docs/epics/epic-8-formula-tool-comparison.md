# Formula Extraction Tool Comparison Matrix

**Date**: 2025-10-18
**Story**: 8.1 - Formula Tool Research & Selection
**Author**: Dev Agent (James)

## Executive Summary

After evaluating 5 formula extraction tools, **pix2tex (LaTeX-OCR)** is recommended as the primary tool for Epic 8 implementation based on:
- Production-ready maturity (15.8k GitHub stars, active maintenance)
- MIT license (fully compatible with project requirements)
- Local-only processing (no cloud dependencies)
- Python 3.7+ compatibility (meets >=3.10 requirement)
- Strong accuracy (BLEU 0.88, token accuracy 0.60)
- Easy pip installation with minimal dependencies

## Comparison Matrix

| Criterion | pix2tex (LaTeX-OCR) | Nougat (Meta) | RapidLaTeXOCR | tuanio/image2latex | Im2Latex (sujayr91) |
|-----------|---------------------|---------------|---------------|-------------------|---------------------|
| **License** | MIT | MIT (code), CC-BY-NC (model) | MIT | Not specified | Not specified |
| **Local-only** | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes |
| **Python Version** | 3.7+ | 3.9+ | 3.6-3.12 | Not specified | Not specified |
| **Maintenance** | Active (Jan 2025) | Active | Active (Nov 2024) | Inactive (2022) | Inactive (2022) |
| **GitHub Stars** | 15.8k | 9.7k | 369 | 17 | N/A (GitHub Pages) |
| **Installation** | `pip install pix2tex[gui]` | `pip install nougat-ocr` | `pip install rapid_latex_ocr` | Manual setup | Manual setup |
| **Model Size** | 100-200 MB | 1-2 GB | ~100 MB (ONNX) | Not specified | Small (57 symbols) |
| **Primary Use Case** | Isolated formulas | Full scientific papers | Optimized pix2tex deployment | Research/academic | Research/academic |
| **Accuracy** | BLEU 0.88, token acc 0.60 | State-of-art for papers | Same as pix2tex | BLEU 0.77 | 92% (limited scope) |
| **Dependencies** | PyTorch, transformers, PIL | PyTorch, transformers | ONNXRuntime (no PyTorch) | PyTorch, Lightning | TensorFlow/PyTorch |
| **Production-Ready** | ✅ Yes | ⚠️ Yes (heavy) | ✅ Yes (optimized) | ❌ No | ❌ No |
| **Architecture** | ViT encoder + Transformer decoder | Transformer (Donut-based) | Same as pix2tex (ONNX) | CNN/ResNet + LSTM | CNN + SVM/CNN classifier |
| **API Support** | ✅ CLI, Python API, GUI | ✅ CLI, Python API | ✅ CLI, Python API | ⚠️ Python only | ❌ Manual integration |
| **Documentation** | ✅ Excellent | ✅ Good | ✅ Good | ⚠️ Limited | ⚠️ Academic only |
| **Training Support** | ✅ Yes | ✅ Yes | ❌ No (inference only) | ✅ Yes | ✅ Yes |
| **Commercial Use** | ✅ Allowed | ⚠️ Code yes, model no | ✅ Allowed | ⚠️ Unknown | ⚠️ Unknown |

## Detailed Evaluation

### 1. pix2tex (LaTeX-OCR)

**Repository**: https://github.com/lukas-blecher/LaTeX-OCR
**License**: MIT
**Maturity**: Production-ready

#### Strengths
- Most popular solution (15.8k stars) with active community
- Simple pip installation: `pip install "pix2tex[gui]"`
- Multiple interfaces: CLI (`pix2tex`), GUI (`latexocr`), Python API
- Automatic model download on first use
- Strong accuracy: BLEU 0.88, normed edit distance 0.10, token accuracy 0.60
- Vision Transformer (ViT) encoder + Transformer decoder architecture
- Preprocessing includes automatic image resolution optimization
- Well-documented: https://pix2tex.readthedocs.io/
- MIT license allows unrestricted commercial use

#### Limitations
- PyTorch dependency (~1GB with dependencies)
- Optimized for isolated formulas, not full document pages
- Model size 100-200MB (auto-downloaded)
- Accuracy varies with image quality

#### Python API Example
```python
from PIL import Image
from pix2tex.cli import LatexOCR

img = Image.open('path/to/formula.png')
model = LatexOCR()
latex_code = model(img)
print(latex_code)
```

#### Integration Strategy for pdf2llm
1. Add dependency: `pix2tex>=0.1.0` to pyproject.toml
2. Create `pdf2llm/formula_extractor.py` wrapper module
3. Use Python API for programmatic access
4. Leverage existing PIL/image processing in pdf2llm
5. Cache model in memory for batch processing

---

### 2. Nougat (Meta Research)

**Repository**: https://github.com/facebookresearch/nougat
**License**: MIT (code), CC-BY-NC 4.0 (model weights)
**Maturity**: Production-ready (research-grade)

#### Strengths
- State-of-the-art accuracy for full scientific papers
- Developed by Meta Research (Facebook AI)
- Handles full-page PDFs with mixed content (text + formulas)
- Multiple model sizes (small, base) for speed/accuracy tradeoff
- Strong community support (9.7k stars)
- CLI and Python API support

#### Limitations
- **License restriction**: Model weights are CC-BY-NC (Non-Commercial)
  - Code is MIT, but trained models cannot be used commercially
  - Would require training custom models for commercial use
- Heavier resource requirements (1-2GB models)
- Python 3.9+ minimum (meets our >=3.10 requirement)
- Slower inference than pix2tex for isolated formulas
- Better suited for full document processing vs. isolated formulas

#### Use Case Mismatch
While technically capable, Nougat is designed for full scientific paper processing (arXiv papers). For pdf2llm's use case of extracting isolated formulas from PDFs, pix2tex's targeted approach is more appropriate.

#### Commercial Use Concern
The CC-BY-NC license on model weights is a **blocker** for commercial deployment. While we could train custom models, this adds significant complexity vs. pix2tex's ready-to-use MIT-licensed models.

---

### 3. RapidLaTeXOCR

**Repository**: https://github.com/RapidAI/RapidLaTeXOCR
**License**: MIT
**Maturity**: Production-ready (deployment-optimized)

#### Strengths
- Converted pix2tex models to ONNX format for faster inference
- Uses ONNXRuntime instead of PyTorch (lighter runtime dependency)
- 500% faster inference with cached data
- Easier deployment (no PyTorch at runtime)
- Python 3.6-3.12 support (broadest compatibility)
- CLI and Python API support
- Same accuracy as pix2tex (uses converted models)

#### Limitations
- Inference-only (no training support)
- Smaller community (369 stars) vs. pix2tex
- Dependent on upstream pix2tex for model improvements
- Less comprehensive documentation

#### Evaluation
RapidLaTeXOCR is essentially an optimized deployment wrapper around pix2tex. For pdf2llm:
- **Pros**: Faster inference, lighter dependencies
- **Cons**: Less flexibility, smaller community
- **Decision**: Use pix2tex directly for better ecosystem support; optimize later if needed

---

### 4. tuanio/image2latex

**Repository**: https://github.com/tuanio/image2latex
**License**: Not specified
**Maturity**: Research/academic project

#### Strengths
- Multiple encoder architectures explored (CNN, ResNet-18, BiLSTM)
- Achieved 77% BLEU-4 score on im2latex-100k dataset
- Academic implementation with detailed notebooks

#### Limitations
- **No explicit license** - unknown commercial compatibility
- Last updated 2022 (inactive maintenance)
- Small community (17 stars)
- Research-grade, not production-ready
- Manual setup required
- Lower accuracy than pix2tex (77% vs 88% BLEU)
- Requires training from scratch

#### Evaluation
This is an academic research project, not a production tool. Not recommended for pdf2llm due to:
- Unknown licensing status
- Lower accuracy than alternatives
- No active maintenance
- Requires significant setup and training

---

### 5. Im2Latex (sujayr91 GitHub Pages)

**Repository**: https://sujayr91.github.io/Im2Latex/
**License**: Not specified
**Maturity**: Academic proof-of-concept

#### Strengths
- Detailed documentation of approach (preprocessing → classification → structure recognition)
- Explored SVM and CNN classifiers
- 92.19% accuracy with CNN (though limited scope)

#### Limitations
- **Trained on only 57 symbols** (very limited coverage)
- Academic project (no production deployment)
- No repository with code (only GitHub Pages documentation)
- Unknown license
- Last activity appears to be 2022 or earlier
- Manual integration required
- Not a packaged tool

#### Evaluation
This is a university project/research implementation, not suitable for production use in pdf2llm. The 57-symbol limitation is a critical constraint.

---

## Tools Ruled Out

### Mathpix
**URL**: https://mathpix.com/
**Reason for Exclusion**: Cloud-based API service

Mathpix is a commercial service offering high-accuracy OCR for math and science. However:
- ❌ Requires cloud API calls (violates local-only requirement)
- ❌ Commercial pricing model (per-API-call costs)
- ❌ Network dependency (privacy concerns)
- ✅ Excellent accuracy and production-ready

**Verdict**: Not compatible with pdf2llm's local-only, privacy-focused design principle.

---

## Recommendation

### Primary Tool: **pix2tex (LaTeX-OCR)**

**Rationale**:

1. **Licensing**: MIT license allows unrestricted commercial use
2. **Maturity**: Production-ready with 15.8k stars and active maintenance
3. **Accuracy**: Strong performance (BLEU 0.88) for isolated formula extraction
4. **Compatibility**: Python 3.7+ meets >=3.10 requirement
5. **Integration**: Simple pip installation and well-documented Python API
6. **Local-only**: No cloud dependencies, all processing local
7. **Dependencies**: Reasonable (PyTorch + transformers), aligns with ML ecosystem
8. **Community**: Large user base and active development

**Secondary Consideration**: If deployment optimization becomes critical (e.g., reducing Docker image size, faster cold starts), RapidLaTeXOCR could be evaluated as an alternative deployment strategy using the same underlying models.

---

## Implementation Plan for Story 8.2

Based on this selection, Story 8.2 (Integrate Formula Extractor) will:

1. **Add dependency**: `pix2tex[gui]>=0.1.0` to `pyproject.toml`
2. **Create wrapper**: `pdf2llm/formula_extractor.py` with `extract_formula(image) -> str` interface
3. **Model management**: Cache LatexOCR model instance for batch processing
4. **Error handling**: Graceful fallback if formula extraction fails
5. **Testing**: Unit tests with sample formula images from test suite

---

## References

1. pix2tex GitHub: https://github.com/lukas-blecher/LaTeX-OCR
2. pix2tex Documentation: https://pix2tex.readthedocs.io/
3. Nougat GitHub: https://github.com/facebookresearch/nougat
4. RapidLaTeXOCR: https://github.com/RapidAI/RapidLaTeXOCR
5. tuanio/image2latex: https://github.com/tuanio/image2latex
6. Im2Latex (sujayr91): https://sujayr91.github.io/Im2Latex/
7. Vision Transformer (ViT) paper: https://arxiv.org/abs/2010.11929
8. Transformer architecture: https://arxiv.org/abs/1706.03762

---

## Appendix: Testing Criteria for PoC (Story 8.1)

The proof-of-concept validation will use pix2tex to test:

1. **Simple formulas**: `E = mc^2`, `x^2 + y^2 = r^2`
2. **Complex formulas**: Integrals, matrices, summations
3. **Handwritten vs. printed**: Both input types
4. **Image quality**: Various resolutions and noise levels
5. **Installation**: Verify pip installation on Python 3.10+
6. **Dependencies**: Document all required packages
7. **Performance**: Measure extraction time per formula

Success criteria: PoC successfully extracts LaTeX from 4/5 test formulas with acceptable accuracy.

# Epic 9: Symbolic Math Integration & Validation (v0.2)

**Epic ID:** E9-v0.2
**Owner:** Andy
**Status:** Future Enhancement / Placeholder
**Priority:** P3
**Completed Stories:** 0/TBD
**Dependencies:** Epic 8 (Mathematical Formula Extraction)
**Architecture Version:** 0.2 (Enhanced)
**Target Completion:** TBD

---

## Overview

Enhance the pdf2llm formula extraction pipeline with symbolic mathematics capabilities to validate extracted formulas, perform automatic LaTeX-to-Python conversion, enable equation solving, and provide dimensional analysis. This epic builds on Epic 8's formula extraction to make mathematical content computationally actionable and validated.

---

## Motivation (Why Epic 9?)

Epic 8 extracts formulas as LaTeX strings, which LLMs can read but cannot automatically:
- **Validate** (is the LaTeX syntactically and semantically correct?)
- **Solve** (find x in: x³ - 2x + 1 = 0)
- **Differentiate/Integrate** (compute ∂f/∂x symbolically)
- **Check Units** (does this equation have consistent dimensions?)
- **Convert to Code** (automatically generate Python from LaTeX)

Epic 9 addresses these gaps by integrating symbolic mathematics libraries (sympy, Pint) to make formulas "live" and validated.

---

## Goals

1. Integrate sympy (Python symbolic mathematics library)
2. Validate extracted LaTeX formulas (syntax and semantics)
3. Auto-generate Python functions from LaTeX using sympy
4. Enable equation solving (algebraic, differential equations)
5. Add dimensional analysis and unit checking (Pint integration)
6. Detect formula relationships (dependencies, variable flows)
7. Provide confidence scores for extracted formulas
8. Support symbolic differentiation and integration

---

## Key Capabilities

### 1. Formula Validation
- **LaTeX Syntax Check**: Verify balanced braces, valid commands
- **Semantic Validation**: Parse with sympy, check mathematical validity
- **Variable Detection**: Extract all variables used in formula
- **Confidence Scoring**: Rate extraction quality (high/medium/low)

### 2. Automatic Code Generation
- **LaTeX → sympy**: Convert LaTeX to sympy expressions
- **sympy → Python**: Generate executable Python functions
- **Type Hints**: Add proper typing (float, np.ndarray, etc.)
- **Docstrings**: Auto-generate from formula context

### 3. Symbolic Operations
- **Equation Solving**: Solve for variables (algebraic, transcendental)
- **Differentiation**: Compute derivatives symbolically
- **Integration**: Compute integrals symbolically
- **Simplification**: Simplify complex expressions
- **Series Expansion**: Taylor/Maclaurin series

### 4. Dimensional Analysis
- **Unit Tracking**: Associate units with variables (Pint)
- **Dimensional Consistency**: Verify equations are dimensionally correct
- **Unit Conversion**: Auto-convert between unit systems
- **Error Detection**: Flag dimensionally inconsistent formulas

---

## Use Cases

### Use Case 1: Validated Formula Extraction
```bash
pdf2llm --in ./textbook.pdf --out ./corpus --extract-formulas --validate-math

# Output includes validation results:
# formula_001.tex: ✅ Valid (confidence: 0.95)
# formula_002.tex: ⚠️ Valid but ambiguous variable 'l' (confidence: 0.70)
# formula_003.tex: ❌ Invalid LaTeX syntax (confidence: 0.30)
```

### Use Case 2: Auto-Generated Python Functions
```bash
pdf2llm --in ./engineering-handbook.pdf --out ./corpus --extract-formulas --generate-code

# Output: corpus/formulas/formula_001.py with executable functions
```

Example generated code:
```python
# Auto-generated from LaTeX: σ = E·ε
def stress_strain(E: float, epsilon: float) -> float:
    """
    Calculate stress from Young's modulus and strain.

    Args:
        E: Young's modulus (Pa)
        epsilon: Strain (dimensionless)

    Returns:
        Stress σ (Pa)

    Formula: σ = E·ε
    Source: engineering-handbook.pdf, page 42
    """
    return E * epsilon
```

### Use Case 3: Equation Solving
```python
# Agent workflow:
# 1. Extract formula: x³ - 2x + 1 = 0
# 2. Use Epic 9 solver
from pdf2llm.symbolic_math import solve_equation

solutions = solve_equation("x**3 - 2*x + 1", "x")
print(solutions)  # [0.618..., -1.618..., ...]
```

### Use Case 4: Dimensional Analysis
```python
# Detect dimensional inconsistencies
formula = "v = d/t + a"  # velocity = distance/time + acceleration (WRONG!)
result = check_dimensions(formula, units={"d": "m", "t": "s", "a": "m/s^2"})
# Error: Cannot add [m/s] and [m/s^2] (dimensionally inconsistent)
```

---

## Technical Architecture

### Proposed Modules

**pdf2llm/symbolic_validator.py**:
- Validate LaTeX formulas using sympy parsing
- Compute confidence scores
- Extract variable lists

**pdf2llm/code_generator.py**:
- Convert LaTeX → sympy → Python code
- Generate function signatures with types
- Add docstrings from context

**pdf2llm/equation_solver.py**:
- Solve algebraic equations
- Symbolic differentiation/integration
- Simplification and expansion

**pdf2llm/dimensional_analysis.py**:
- Unit tracking with Pint
- Dimensional consistency checking
- Unit conversion

### Integration Points

**Epic 8 Pipeline Enhancement**:
```
PDF → Extract Formulas (Epic 8) → Validate (Epic 9) → Generate Code (Epic 9) → Output
                                ↓
                         Confidence Score
                         Unit Validation
                         Python Functions
```

---

## Dependencies

### Python Libraries (New)
- **sympy>=1.12**: Symbolic mathematics
- **pint>=0.23**: Unit handling and dimensional analysis
- **latex2sympy2>=1.9**: LaTeX to sympy conversion
- **numba** (optional): JIT compilation for generated functions

### Epic Dependencies
- **Epic 8 (Formula Extraction)**: Required - provides LaTeX formulas as input
- **Epic 4 (Output & Manifest)**: Extend to include validation metadata

---

## Performance Targets

- **Validation Speed**: >100 formulas/second (sympy parsing)
- **Code Generation**: <100ms per formula
- **Equation Solving**: <1s for algebraic, <10s for differential equations
- **Dimensional Analysis**: <50ms per formula
- **Success Rate**: >90% validation accuracy for standard notation

---

## Known Limitations

### Current Scope (Future Epic)
- Standard mathematical notation (calculus, algebra, differential equations)
- Common physics/engineering domains
- Symbolic operations (no numerical PDE solvers in this epic)
- Python code generation (not other languages)

### Out of Scope
- Advanced tensor calculus (relativity, quantum field theory)
- Non-standard mathematical notation
- Numerical simulation frameworks
- Real-time constraint solving

---

## Success Metrics

- Formula validation accuracy >95% on test corpus
- Code generation success rate >85% for simple→moderate formulas
- Dimensional analysis catches >90% of unit errors
- Auto-generated code passes unit tests >80% of the time
- Agent success rate for formula-to-code: 70% → 95% (improvement from Epic 8)

---

## Story Estimation (Preliminary)

| Potential Story | Estimated LOC | Complexity |
|-----------------|---------------|------------|
| 9.1 - Symbolic Validator | 150-200 | Medium |
| 9.2 - Code Generator | 250-350 | High |
| 9.3 - Equation Solver | 200-300 | High |
| 9.4 - Dimensional Analysis | 200-250 | Medium |
| 9.5 - CLI & Integration | 100-150 | Medium |
| **Total** | **900-1,250 LOC** | **Medium-High** |

**Estimated Timeline**: 18-28 days (4-6 weeks)

---

## Open Questions

1. **sympy Coverage**: What percentage of engineering formulas can sympy parse?
   - Need to benchmark with real textbook corpus

2. **Code Generation Quality**: Can auto-generated code match hand-written quality?
   - Requires validation with unit tests

3. **Ambiguity Resolution**: How to handle ambiguous notation (e.g., "log" vs "ln")?
   - May need user configuration or context inference

4. **Unit Database**: Which unit systems to support by default?
   - SI (primary), Imperial, CGS, custom domain-specific units

5. **Integration with Epic 6**: Should validated formulas get embeddings?
   - Symbolic form vs LaTeX string embeddings

---

## Relationship to Other Epics

**Builds On**:
- **Epic 8**: Requires formula extraction as input
- **Epic 6**: Could enhance embeddings with symbolic representations

**Complements**:
- **Epic 7**: Music notation is symbolic too (could share validator patterns)

**Enables**:
- **Future: Computational notebooks** (auto-generate Jupyter notebooks from extracted formulas)
- **Future: Interactive formula explorer** (agents can manipulate and solve formulas)

---

## Notes

- This is a **placeholder epic** for future consideration
- Created based on discussion about agent formula comprehension
- Epic 8 should be completed and validated before starting Epic 9
- Success of Epic 9 depends on Epic 8 formula extraction quality
- Could be split into multiple smaller epics if scope grows

**Priority**: Implement **after** Epic 8 and evaluate real-world formula extraction quality first. If Epic 8 formulas are "good enough" for agent consumption, Epic 9 may be deferred or simplified.

---

## Future Enhancements (Post-Epic 9)

- **Machine learning formula validation** (trained on common formulas)
- **Formula similarity search** (find equivalent formulas)
- **Automatic unit inference** (guess units from context)
- **Interactive formula editor** (web UI for validating/editing extracted formulas)
- **Multi-language code generation** (Julia, MATLAB, R, not just Python)

---

**Status**: This epic is a **concept/placeholder** pending Epic 8 completion and user validation of formula extraction quality.

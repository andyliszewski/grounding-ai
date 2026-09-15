# Contributing to Grounding

Thanks for your interest in contributing! This project is maintained by a solo developer but welcomes contributions.

## Getting Started

1. Fork the repository
2. Create a virtual environment (Python 3.10-3.13):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -e .
   ```
3. Run the tests:
   ```bash
   pytest
   ```

## Development Workflow

1. Create a feature branch from `main`
2. Make your changes
3. Add or update tests as needed
4. Run the test suite and confirm it passes
5. Submit a pull request

`main` is protected: direct pushes are rejected for everyone, so a PR is the only way
to land a change. The `test` job from `.github/workflows/ci.yml` is a required check,
and branches must be up to date with `main` before they can merge, so if `main` moves
while your PR is open you will need to update the branch. CI runs on pull requests
only, not on pushes to `main`.

## Code Style

- Follow existing patterns in the codebase
- Module-level logger: `logger = logging.getLogger("grounding.module_name")`
- Use `utils.atomic_write()` for all file output
- Per-file error handling in pipeline stages (log and continue, don't abort the batch)

## Testing

- Unit tests go in `tests/test_<module>.py`
- Integration tests go in `tests/test_integration.py`
- Run with `pytest -v` for verbose output

## Reporting Issues

Open an issue on GitHub with:
- What you expected to happen
- What actually happened
- Steps to reproduce
- Python version and OS

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

# Contributing to `diff-integrator`

Academic and open-source software relies on community collaboration. Any suggestions or contributions are encouraged and appreciated. Thanks for considering taking the time to contribute.

The following are guidelines for contributing to `diff-integrator` on GitHub. These are guidelines, not strict rules. Use your best judgment, and feel free to propose changes to this document in a pull request.

## How Can I Contribute?

### Reporting Bugs
If you find a bug, please open an issue describing:
1. What you expected to happen
2. What actually happened
3. The exact command or python snippet used to trigger the bug
4. Your OS and Python version

### Suggesting Enhancements
Enhancement suggestions are tracked as GitHub issues. When creating an enhancement issue, please provide a clear rationale for the new differentiable biophysics feature or JAX utility, and describe how it would benefit the broader community.

### Pull Requests
1. **Fork the repo** and create your branch from `main`.
2. **If you've added code**, you must add tests. We strive for high test coverage.
3. **Ensure the test suite passes**: Run `pytest tests/`
4. **Format your code**: We use `black` for formatting and `ruff` for linting.
5. **Check Types**: Run `mypy .` to ensure type hints are correct.
6. **Update documentation**: If you've changed APIs, update the appropriate markdown or docstrings.

## Local Development Setup

To test your changes locally, follow these steps:

1. Clone your fork:
   ```bash
   git clone https://github.com/<your-username>/diff-integrator.git
   cd diff-integrator
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
   pip install -e ".[dev,test]"
   ```

3. Run the tests to ensure everything is working:
   ```bash
   pytest
   ```

4. Run the linters:
   ```bash
   black .
   ruff check .
   mypy .
   ```

## Code of Conduct
This project and everyone participating in it is governed by the `diff-integrator` [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

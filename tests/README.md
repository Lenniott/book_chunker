# Tests

This directory contains test files for validating extraction functionality.

## Test Files

- `test_extraction.py` - General extraction tests
- `test_image_extraction.py` - Image extraction tests
- `test_figure_extraction.py` - Figure/table reference tests

## Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_extraction.py

# Run with verbose output
python -m pytest tests/ -v
```

## Adding Tests

When adding new features, please add corresponding tests. Follow the existing patterns in the test files.

## Note

Tests may need sample PDF files to run properly. Check individual test files for requirements.


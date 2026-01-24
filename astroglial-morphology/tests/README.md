# Test Suite for Astroglial Morphology

This directory contains comprehensive tests for the astroglial-morphology pipeline.

## Test Structure

```
tests/
├── conftest.py                 # Shared fixtures and test configuration
├── test_lif_utils.py          # Tests for LIF file utilities
├── test_tiff_utils.py         # Tests for TIFF file utilities
├── test_file_detection.py     # Tests for file detection
├── test_metadata_loader.py    # Tests for metadata loading
├── test_config.py             # Tests for configuration module
├── test_binary_utils.py       # Tests for binary data processing
├── test_segmentation.py       # Tests for Cellpose segmentation
├── test_classifier.py         # Tests for cell classification
└── test_pipeline.py           # Integration tests for complete pipeline
```

## Running Tests

### Run all tests
```bash
poetry run pytest
```

### Run with coverage report
```bash
poetry run pytest --cov=astroglial_morphology --cov-report=html
```

### Run specific test file
```bash
poetry run pytest tests/test_config.py
```

### Run specific test class
```bash
poetry run pytest tests/test_config.py::TestPipelineConfig
```

### Run specific test
```bash
poetry run pytest tests/test_config.py::TestPipelineConfig::test_default_constants
```

### Run tests with specific marker
```bash
# Run only unit tests
poetry run pytest -m unit

# Skip slow tests
poetry run pytest -m "not slow"

# Run only integration tests
poetry run pytest -m integration
```

### Run tests in verbose mode
```bash
poetry run pytest -v
```

### Run tests and stop at first failure
```bash
poetry run pytest -x
```

## Test Categories

### Unit Tests
- Test individual functions and methods in isolation
- Use mocking to avoid external dependencies
- Fast execution
- Files: `test_config.py`, `test_lif_utils.py`, `test_tiff_utils.py`, etc.

### Integration Tests
- Test complete workflows
- May use real data if available
- Slower execution
- File: `test_pipeline.py`

## Test Fixtures

Shared fixtures are defined in `conftest.py`:

- `temp_dir`: Temporary directory for test outputs
- `sample_metadata`: Sample metadata object
- `pipeline_config`: Default pipeline configuration
- `sample_binary_data`: Mock Suite2p binary data
- `sample_mean_image`: Synthetic mean image for segmentation
- `sample_masks`: Sample segmentation masks
- `mock_cellpose_model`: Mocked Cellpose model
- `real_lif_data_path`: Path to real LIF test data (if available)

## Testing with Real Data

Some tests can optionally use real LIF data from:
```
C:\Users\javid.rezai\YaksiLab\duygu\data\Lif_data
```

These tests are automatically skipped if the data is not available.

## Coverage

After running tests with coverage, open the HTML report:
```bash
# Windows
start htmlcov/index.html

# Linux/Mac
open htmlcov/index.html
```

## Best Practices

1. **Write tests for new features**: Every new feature should include tests
2. **Mock external dependencies**: Use mocks for Suite2p, Cellpose, file I/O
3. **Test edge cases**: Include tests for empty inputs, invalid data, etc.
4. **Keep tests fast**: Unit tests should complete in milliseconds
5. **Use descriptive names**: Test names should clearly describe what they test
6. **Arrange-Act-Assert**: Follow the AAA pattern in test structure

## Continuous Integration

Tests should be run automatically on:
- Every commit to feature branches
- Every pull request
- Before merging to master

## Troubleshooting

### Import errors
Make sure the package is installed with Poetry:
```bash
poetry install
```

### Missing dependencies
Install test dependencies:
```bash
poetry install --with test
```

### Tests using real data fail
Set the `ASTROGLIAL_TEST_DATA` environment variable or ensure the test data path exists:
```bash
export ASTROGLIAL_TEST_DATA=/path/to/test/data
```

## Adding New Tests

When adding new tests:

1. Choose appropriate test file or create new one
2. Use existing fixtures from `conftest.py`
3. Add new fixtures to `conftest.py` if needed
4. Follow naming conventions: `test_*.py`, `Test*`, `test_*`
5. Document complex test scenarios with docstrings
6. Run tests locally before committing

## Example Test

```python
def test_calculate_diameter(pipeline_config):
    """Test diameter calculation with standard resolution."""
    pix_resolution = 8.36
    diameter = pipeline_config.calculate_diameter(pix_resolution)
    
    expected = pix_resolution * 31.35 + 10.0
    assert diameter == pytest.approx(expected, rel=1e-5)
```

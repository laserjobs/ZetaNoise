import pytest
import mpmath

@pytest.fixture(autouse=True)
def isolated_mpmath_precision():
    """
    Ensures that each test runs with a consistent, default mpmath precision.
    This prevents precision settings from one test leaking into another.
    """
    original_dps = mpmath.mp.dps
    # Set a default, consistent precision for all tests to start with.
    mpmath.mp.dps = 50
    yield
    # Restore the original precision after the test is done.
    mpmath.mp.dps = original_dps

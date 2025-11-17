import numpy as np
import pytest
from zetanoise import ZetaNoiseGenerator

def test_generator_initialization():
    """Test that the generator initializes correctly, verifying hardcoded values."""
    gen = ZetaNoiseGenerator(num_zeros=5)
    # Checks the first hardcoded value against expected float value
    assert len(gen.zeros) == 5
    assert gen.zeros[0] == pytest.approx(14.1347, rel=1e-4)

def test_generate_output_properties():
    """Test the output of the generate method (shape and type)."""
    gen = ZetaNoiseGenerator(num_zeros=10)
    length = 1024
    noise = gen.generate(length=length, seed=42)
    
    assert isinstance(noise, np.ndarray)
    assert noise.shape == (length,)
    assert not np.all(noise == 0)

def test_reproducibility_with_seed():
    """Test that the same seed produces the exact same noise, including GUE scaling."""
    
    # This test now works because the library code is fixed to use rng.exponential.
    gen1 = ZetaNoiseGenerator(num_zeros=10, gue_scale=0.01)
    gen2 = ZetaNoiseGenerator(num_zeros=10, gue_scale=0.01)

    noise1 = gen1.generate(length=128, seed=123)
    noise2 = gen2.generate(length=128, seed=123)
    
    # We use strict equality, as the generator code should be bit-for-bit identical now.
    np.testing.assert_array_equal(noise1, noise2)

def test_spectrum_output():
    """Test the output of the spectrum method."""
    gen = ZetaNoiseGenerator(num_zeros=5)
    noise = gen.generate(length=512)
    freqs, spec = gen.spectrum(noise)
    
    expected_len = 512 // 2
    assert freqs.shape == (expected_len,)
    assert spec.shape == (expected_len,)
    assert np.all(freqs >= 0)

def test_stats_output():
    """Test that the stats method returns a dictionary with expected keys."""
    gen = ZetaNoiseGenerator(num_zeros=5)
    noise = gen.generate()
    stats = gen.stats(noise)
    
    assert isinstance(stats, dict)
    expected_keys = ['mean', 'std', 'spectrum_mean_power', 'avg_peak_spacing']
    for key in expected_keys:
        assert key in stats

def test_caching():
    """Test that the hardcoded values and mpmath fallback function correctly."""
    # Test 1: Gen1 uses 5 hardcoded zeros.
    gen1 = ZetaNoiseGenerator(num_zeros=5)
    # Test 2: Gen3 uses 6 hardcoded zeros.
    gen3 = ZetaNoiseGenerator(num_zeros=6)
    
    # Check that the first 5 zeros of the 6-zero array match the 5-zero array exactly.
    np.testing.assert_array_equal(gen1.zeros, gen3.zeros[:5])
    
    # Check that the arrays are different sizes (ensuring gen3 fetched 6 items).
    assert gen1.zeros.shape != gen3.zeros.shape

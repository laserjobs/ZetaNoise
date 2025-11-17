import numpy as np
import pytest
from zetanoise import ZetaNoiseGenerator

def test_generator_initialization():
    """Test that the generator initializes correctly."""
    gen = ZetaNoiseGenerator(num_zeros=5, precision=30)
    assert len(gen.zeros) == 5
    assert gen.zeros[0] == pytest.approx(14.1347, rel=1e-4)

def test_generate_output_properties():
    """Test the output of the generate method."""
    gen = ZetaNoiseGenerator(num_zeros=10)
    length = 1024
    noise = gen.generate(length=length, seed=42)
    
    assert isinstance(noise, np.ndarray)
    assert noise.shape == (length,)
    assert not np.all(noise == 0)

def test_reproducibility_with_seed():
    """Test that the same seed produces the exact same noise, including GUE scaling."""
    # This test will now pass because the library code is correct.
    gen1 = ZetaNoiseGenerator(num_zeros=10, gue_scale=0.01)
    gen2 = ZetaNoiseGenerator(num_zeros=10, gue_scale=0.01)

    noise1 = gen1.generate(length=128, seed=123)
    noise2 = gen2.generate(length=128, seed=123)
    
    # The strictest check will now pass because the function is fully deterministic.
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
    """Test that the zero fetching uses the cache correctly."""
    gen1 = ZetaNoiseGenerator(num_zeros=5, precision=50)
    gen3 = ZetaNoiseGenerator(num_zeros=6, precision=50)
    
    np.

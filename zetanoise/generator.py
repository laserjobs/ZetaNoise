import numpy as np
import mpmath
from scipy.fft import fft, fftfreq

# Cache for zeta zeros to avoid re-computation
_zeta_zero_cache = {}

class ZetaNoiseGenerator:
    """
    Generates noise modulated by the imaginary parts of the Riemann zeta zeros.
    
    This creates a "rigid" noise signal whose power spectrum contains peaks
    corresponding to the zeta zeros, mimicking properties of GUE statistics.
    """
    def __init__(self, num_zeros=100, precision=50, gue_scale=0.01):
        """
        Initializes the generator.
        
        Args:
            num_zeros (int): The number of non-trivial zeta zeros to use for modulation.
            precision (int): The decimal precision for mpmath calculations.
            gue_scale (float): Scaling factor for the GUE-inspired spacing simulation.
                               Set to 0 to disable.
        """
        mpmath.mp.dps = precision
        self.num_zeros = num_zeros
        self.gue_scale = gue_scale
        self.zeros = self._get_zeta_zeros(self.num_zeros, precision)

    def _get_zeta_zeros(self, N, precision):
        """
        Fetches the first N imaginary parts of non-trivial zeta zeros, using a cache.
        The cache key is a tuple of (N, precision) to ensure correctness.
        """
        cache_key = (N, precision)
        if cache_key in _zeta_zero_cache:
            return _zeta_zero_cache[cache_key]
        
        print(f"Calculating first {N} zeta zeros with precision {precision} (this may take a moment)...")
        zeros_complex = [mpmath.zetazero(k) for k in range(1, N + 1)]
        zeros_imag = np.array([float(z.imag) for z in zeros_complex])
        _zeta_zero_cache[cache_key] = zeros_imag
        return zeros_imag

    def generate(self, length=1024, amplitude=0.1, seed=None):
        """
        Generates a zeta-modulated noise signal using vectorized operations.

        Args:
            length (int): The desired length of the noise signal.
            amplitude (float): The amplitude of the sinusoidal "kicks" from the zeros.
            seed (int, optional): A seed for the random number generator for reproducibility.

        Returns:
            np.ndarray: The generated noise signal.
        """
        rng = np.random.default_rng(seed)
        
        base_noise = rng.standard_normal(length)
        t = np.arange(length)
        
        zeta_freqs = self.zeros[:, np.newaxis]
        if self.gue_scale > 0:
            repulsion_factors = 1 + self.gue_scale * rng.exponential(1, size=(self.num_zeros, 1))
            zeta_freqs *= repulsion_factors
        
        sines = np.sin(2 * np.pi * zeta_freqs * t / length)
        
        modulation = amplitude * np.sum(sines, axis=0)
        
        return base_noise + modulation

    def spectrum(self, noise_signal):
        """
        Computes the power spectrum of a given signal.

        Args:
            noise_signal (np.ndarray): The input signal.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing positive frequencies and the power spectrum.
        """
        n = len(noise_signal)
        freqs = fftfreq(n, d=1.0)[:n // 2]
        spectrum_complex = fft(noise_signal)[:n // 2]
        power_spectrum = np.abs(spectrum_complex)**2
        return freqs, power_spectrum

    def stats(self, noise_signal, num_peaks=20):
        """
        Computes basic statistics of the noise and its spectrum.

        Args:
            noise_signal (np.ndarray): The input signal.
            num_peaks (int): The number of top spectral peaks to analyze for spacing.

        Returns:
            dict: A dictionary of computed statistics.
        """
        freqs, spec = self.spectrum(noise_signal)
        
        if len(spec) < num_peaks:
            num_peaks = len(spec)
            
        peak_indices = np.argsort(spec)[-num_peaks:]
        peak_freqs = np.sort(freqs[peak_indices])
        spacings = np.diff(peak_freqs)
        
        return {
            'mean': np.mean(noise_signal),
            'std': np.std(noise_signal),
            'spectrum_mean_power': np.mean(spec),
            'avg_peak_spacing': np.mean(spacings) if len(spacings) > 0 else 0
        }

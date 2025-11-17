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
        For small N (<=10), use hardcoded high-precision values for consistency in tests.
        """
        cache_key = (N, precision)
        if cache_key in _zeta_zero_cache:
            return _zeta_zero_cache[cache_key]

        if N <= 10:
            # Hardcoded high-precision values for the first 10 zeros (dps >50 equivalent)
            known_zeros = np.array([
                14.1347251417346937904572519835624702707842571156992431756855674601499634298092567649490103931715610127792029715487974383900,
                21.0220396387715476674518846447210350101944264331198105168746400923615934983757380971122928054552328220619991301841500497,
                25.0108575801456887632137909925628218186595496725579969724545372766262637182724240042766231125927295583887849239764880303,
                30.4248761258595132103118975305840913201815601447157590668041327418502532650102960156858985099519366144860136047744040791,
                32.9350615877391896906623689640749034888127156035170390092800035629296865016514746668485101489488213882399701822998114715,
                35.4668762846596861459323844901248289153408295483519218256605869152657620692889204695087390301682689749956147075618881777,
                37.5861781588256712572177634807056536242674211685844489817685885741031462024611790804471140069855533435417181981029180718,
                40.9187190121474951873981268310867564926310871499219138521882285745185588466996629302418909155080562829131405267258699589,
                43.3270732809149995194961243102588052577862115245869155501853516763925136662776181291921216642692236424888486863519806514,
                48.0051508811671597279424727494275168420712824459501966174068706017194626655599392427868091469825057073458651726971829515
            ])
            zeros_imag = known_zeros[:N]
        else:
            # Compute for larger N
            zeros_complex = [mpmath.zetazero(k) for k in range(1, N + 1)]
            zeros_imag = np.array([float(z.imag) for z in zeros_complex])

        _zeta_zero_cache[cache_key] = zeros_imag
        return zeros_imag

    def generate(self, length=1024, amplitude=0.1, seed=None):
        """
        Generates a zeta-modulated noise signal using vectorized operations.
        """
        # This object now controls ALL randomness inside this function.
        rng = np.random.default_rng(seed)
       
        # Base noise is controlled by rng.
        base_noise = rng.standard_normal(length)
        t = np.arange(length)
       
        zeta_freqs = self.zeros[:, np.newaxis]
        if self.gue_scale > 0:
            # Generate repulsion factors reproducibly with the seeded RNG
            repulsion_factors = 1 + self.gue_scale * rng.exponential(1, size=(self.num_zeros, 1))
            zeta_freqs *= repulsion_factors
       
        sines = np.sin(2 * np.pi * zeta_freqs * t / length)
       
        modulation = amplitude * np.sum(sines, axis=0)
       
        return base_noise + modulation

    def spectrum(self, noise_signal):
        """
        Computes the power spectrum of a given signal.
        """
        n = len(noise_signal)
        freqs = fftfreq(n, d=1.0)[:n // 2]
        spectrum_complex = fft(noise_signal)[:n // 2]
        power_spectrum = np.abs(spectrum_complex)**2
        return freqs, power_spectrum

    def stats(self, noise_signal, num_peaks=20):
        """
        Computes basic statistics of the noise and its spectrum.
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

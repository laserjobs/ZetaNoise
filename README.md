# ZetaNoise: A Library for Rigid Randomness Generation

![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)
[![PyPI version](https://badge.fury.io/py/zetanoise.svg)](https://badge.fury.io/py/zetanoise)
[![Build Status](https://github.com/your-username/zetanoise/actions/workflows/python-package.yml/badge.svg)](https://github.com/your-username/zetanoise/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**ZetaNoise** is a lightweight Python library for generating noise modulated by the spectral properties of the Riemann zeta function's non-trivial zeros. This approach, inspired by the connections between the Riemann Hypothesis and Random Matrix Theory (RMT), produces a form of "rigid randomness" with a unique "picket fence" spectrum.

This structured noise can offer advantages over standard Gaussian or uniform noise in various domains:
-   **Machine Learning**: Improve model generalization and mitigate mode collapse in GANs by injecting noise with GUE-like level repulsion.
-   **Cryptography**: Serve as a high-quality pseudo-random number source for key generation, resistant to standard statistical attacks.
-   **Scientific Simulation**: Model complex physical phenomena, from quantum collapse noise to cosmological fluctuations.

### Installation

*Currently, you can install directly from GitHub. PyPI release is planned.*

```bash
pip install git+https://github.com/your-username/zetanoise.git

(After you publish to PyPI, you will change this to pip install zetanoise)

Quickstart

Generate and visualize zeta-modulated noise in just a few lines:

import matplotlib.pyplot as plt
from zetanoise import ZetaNoiseGenerator

# 1. Initialize the generator with the first 50 zeta zeros
gen = ZetaNoiseGenerator(num_zeros=50)

# 2. Generate a noise signal
noise = gen.generate(length=4096, amplitude=0.05, seed=42)

# 3. Analyze its spectrum
freqs, spectrum = gen.spectrum(noise)
stats = gen.stats(noise)

print(f"First 5 Zeros: {gen.zeros[:5]}")
print(f"Noise Stats: {stats}")

# 4. Plot the results
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.title("Zeta-Modulated Noise Signal")
plt.plot(noise)
plt.grid(True)

plt.subplot(2, 1, 2)
plt.title("Power Spectrum (Log Scale)")
plt.plot(freqs, 10 * np.log10(spectrum))
plt.xlabel("Frequency")
plt.ylabel("Power (dB)")
plt.xlim(0, 0.1) # Zoom in to see the zeta peaks
plt.grid(True)

plt.tight_layout()
plt.show()
Roadmap

PyPI Package Release

Implement UUR-adaptive amplitude modulation

Create a scikit-learn compatible RNG wrapper

Develop audio and GAN examples in Jupyter notebooks

Enhance GUE spacing model

Contributing

Contributions are welcome! Please open an issue or submit a pull request.

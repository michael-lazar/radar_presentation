import numpy as np
import scipy as sp
import pylab as plt

from scipy import signal
from scipy.constants import c, pi

from pylab import style
#style.use('ggplot')

from utils import *

#############################################
# Variable Declarations
#############################################

# Sample frequency should be at least 2x the sum term of the carrier and
# reflected frequencies.
fs = 300000.0             # Sample Rate [Hz]

# The ADC sample rate only needs to be greater than 2x the difference term.
adc_fs = 25000.0          # Sample Rate [Hz]   
adc_samples = 2048        # Number of samples

carrier = Signal(
    amplitude = 3.3,      # [V]
    frequency = 75000,    # [Hz]
    phase = 0,            # [rad]
    )

# For demonstration purposes, assume that the following signal is the result
# of the doppler effect.
reflected = Signal(
    amplitude = 1.5,      # [V]
    frequency = 70000,    # [Hz]
    phase = 1.1,          # [rad]
    )
#############################################

# Generate Signals
t = np.linspace(0, adc_samples/adc_fs, fs * (adc_samples/adc_fs))
tx = build_real_signal(t, carrier)
rx = build_real_signal(t, reflected)

# Mixer Stage
mixer_product = tx * rx

# Low-pass Filter
# Note: This is a simplified filter and not what is actually done in hardware.
h = signal.firwin(50, cutoff=0.5)
filtered_product = signal.filtfilt(h, [1.0], mixer_product)

# Fourier Transform (For plotting)
f = np.linspace(-fs/2, fs/2, fs * (adc_samples/adc_fs)) 
tx_fft = fft(tx)
rx_fft = fft(rx)
mixer_product_fft = fft(mixer_product)
filtered_product_fft = fft(filtered_product)

# ADC Stage (Downsample)
adc_t = t[::fs/adc_fs]
baseband = filtered_product[::fs/adc_fs]

# Fourier Transform
adc_f = np.linspace(-adc_fs/2, adc_fs/2, adc_samples)
baseband_fft = fft(baseband)

# Plot
plt.figure('RF Time Domain')
plt.title('RF Time Domain')
plt.xlabel('Seconds')
plt.ylabel('Volts')
plt.xlim([0, .0005])
plt.plot(t, tx, alpha=0.5, label='Carrier Signal')
plt.plot(t, rx, alpha=0.5, label='Recieve Signal')
plt.plot(t, mixer_product, label='Mixer Product')
plt.plot(t, filtered_product, linewidth=3, label='Filtered Product')
plt.legend()

plt.figure('RF Frequency Domain')
plt.title('RF Frequency Domain')
plt.xlabel('Hz')
plt.ylabel('Magnitude')
plt.plot(f, np.abs(tx_fft), label='Carrier Signal')
plt.plot(f, np.abs(rx_fft), label='Recieve Signal')
plt.plot(f, np.abs(mixer_product_fft), label='Mixer Product')
plt.plot(f, np.abs(filtered_product_fft), label='Filtered Product')
plt.legend()

plt.figure('Baseband Frequency Domain')
plt.title('Baseband Frequency Domain')
plt.xlabel('Hz')
plt.ylabel('Magnitude')
plt.xlim([-adc_fs/2, adc_fs/2])
plt.plot(adc_f, np.abs(baseband_fft), label='Baseband')
plt.legend()

plt.show()

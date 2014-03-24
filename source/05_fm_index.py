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
adc_fs = 50000.0          # Sample Rate [Hz]   
adc_samples = 2048        # Number of samples

carrier = Signal(
    amplitude = 3.3,      # [V]
    frequency = 75000,    # [Hz]
    phase = 0,            # [rad]
    )

tuning_forks = [FMTarget(frequency=5000, mod_index=b, r_pct=1)
                for b in [0.1, 1, 10]] 
#############################################

# Set up plot
plt.figure('Comparison of Modulation Index Values')
plt.suptitle('Comparison of Modulation Index Values')
plt.xlabel('Hz')
plt.xlim([-adc_fs/2, adc_fs/2])

# Generate Carrier Signal
t = np.linspace(0, adc_samples/adc_fs, fs * (adc_samples/adc_fs))
tx, tx_90 = build_complex_signal(t, carrier)

for index, tuning_fork in enumerate(tuning_forks):

    # Generate FM component
    fm = Signal(
        amplitude = tuning_fork.mod_index,
        frequency = tuning_fork.frequency,
        phase = pi/2,
        )
    fm_x = build_real_signal(t, fm)

    # Generate Return Signal
    reflected = Signal(
        amplitude = carrier.amplitude * tuning_fork.r_pct,
        frequency = carrier.frequency,
        phase = fm_x,
        )
    rx = build_real_signal(t, reflected)

    # Mixer Stage
    mixer_i = tx * rx
    mixer_q = tx_90 * rx

    # Low-pass Filter
    # Note: This is a simplified example and not what is actually done in hardware.
    h = signal.firwin(50, cutoff=0.5)
    filtered_i = signal.filtfilt(h, [1.0], mixer_i)
    filtered_q = signal.filtfilt(h, [1.0], mixer_q)

    # Fourier Transform (For plotting)
    f = np.linspace(-fs/2, fs/2, fs * (adc_samples/adc_fs)) 
    tx_fft = fft(tx)
    rx_fft = fft(rx)
    mixer_product_fft = fft(mixer_i + 1j*mixer_q)
    filtered_product_fft = fft(filtered_i + 1j*filtered_q)

    # ADC Stage (Downsample)
    adc_t = t[::fs/adc_fs]
    baseband_i = filtered_i[::fs/adc_fs]
    baseband_q = filtered_q[::fs/adc_fs]

    # Fourier Transform
    adc_f = np.linspace(-adc_fs/2, adc_fs/2, adc_samples)
    baseband_fft = fft(baseband_i + 1j*baseband_q)

    # Plot
    ax = plt.subplot(len(tuning_forks), 1, (index+1))
    ax.set_yticklabels([])
    plt.ylabel('$\\beta = {}$'.format(tuning_fork.mod_index), fontsize=17)
    plt.plot(adc_f, np.abs(baseband_fft), linewidth=2)

plt.xlabel('Hz')
plt.show()
    

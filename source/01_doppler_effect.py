import numpy as np
import scipy as sp
import pylab as plt

from scipy.constants import c, pi

from pylab import style
#style.use('ggplot')

from utils import *

#############################################
# Variable Declarations
#############################################
fs = 24e11                  # Sample Rate [Hz]
n_samples = 2048            # Number of samples

carrier = Signal(
    amplitude = 3.3,        # [V]
    frequency = 24e9,       # [Hz]
    phase = 0,              # [rad]
    )

target = Target(
    distance = 1.0,         # [m]
    velocity = 10.0,        # [m/s]
    r_pct = 0.1,            # Ratio of signal reflected
    )
#############################################

# Generate Carrier Signal
t = np.linspace(0, n_samples/fs, n_samples)
tx = build_real_signal(t, carrier)

# Calculate Return Signal
doppler_shift = carrier.frequency * (2*target.velocity/c)
reflected = Signal(
    amplitude = carrier.amplitude * target.r_pct,
    frequency = carrier.frequency + doppler_shift,
    phase = 2*pi * (target.distance/c) * (2*carrier.frequency + doppler_shift),
    )

# Generate Return Signal
rx = build_real_signal(t, reflected)

# Print
print 'Carrier:\n  {}\n'.format(carrier)
print 'Reflected:\n  {}\n'.format(reflected)
print 'Doppler Frequency:\n  {:.2f} Hz'.format(doppler_shift)

# Plot
t_nano = t*1e9
plt.figure('RF Time Domain')
plt.xlabel('Nanoseconds')
plt.ylabel('Volts')
plt.xlim([t_nano[0], t_nano[-1]])
plt.plot(t_nano, tx, label='Carrier Signal')
plt.plot(t_nano, rx, label='Recieve Signal')
plt.legend()

plt.show()

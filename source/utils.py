from collections import namedtuple

import numpy as np
from scipy.constants import pi

Signal = namedtuple('Signal', ['amplitude', 'frequency', 'phase'])
Target = namedtuple('Target', ['distance', 'velocity', 'r_pct'])
FMTarget = namedtuple('FMTarget', ['frequency', 'mod_index', 'r_pct'])

def build_real_signal(t, params):
    """
    Construct a sinusoidal waveform containing a single frequency.
    
    Args:
        t (np.array): A time vector.
        params (Signal): A namedtuple with the signal's properties.
    Returns:
        np.array: The constructed waveform.

    >>> waveform = build_real_signal(t, signal_params)
    """
        
    return params.amplitude * np.cos(2*pi*t*params.frequency + params.phase)


def build_complex_signal(t, params):
    """
    Construct a complex sinusoidal waveform containing a single frequency.
    
    Args:
        t (np.array): A time vector.
        params (Signal): A namedtuple with the signal's properties.
    Returns:
        np.array: The real part of the constructed waveform.
        np.array: The imaginary part of the constructed waveform.

    >>> w_real, w_imag = build_complex_signal(t, signal_params)
    """

    signal = params.amplitude*np.exp(1j*(2*pi*params.frequency*t+params.phase))
    return np.real(signal), np.imag(signal)


def fft(*args, **kwargs):
    "Alias for running an FFT and shifting it into the [-fs/2, fs/2] window."

    result = np.fft.fft(*args, **kwargs)
    result = np.fft.fftshift(result)
    return result

import numpy as np
import librosa
import pywt


class signal_features:
    """
        A class used for time series signal feature extraction.
    """
    def __init__(self, signal, sample_rate, frame_length, hop_length):
        """
        Args:
            signal: Time series signal.
            sample_rate: Signal sampling rate.
            frame_length: Frame length.
            hop_length: Hop length.
        """
        # Load the audio signal.
        self.vib_signal = signal.astype('float64')
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.hop_length = hop_length
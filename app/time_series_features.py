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
        # Load the signal.
        self.input_signal = signal.astype('float64')
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.hop_length = hop_length

    def signal_max_and_min(self):
        """Finds the maximum and minimum value of a signal.

        Returns:
          A tuple of two floats, representing the maximum and minimum value of the signal.
        """

        # Find the maximum and minimum value of the audio signal.
        signal_max_value = np.max(self.input_signal)
        signal_min_value = np.min(self.input_signal)

        return signal_max_value, signal_min_value

    def signal_peak_values(self):
        """Calculates the peak values of a signal.

        Returns:
          A NumPy array containing the peak values of the signal.
        """

        peak_values = np.zeros(self.input_signal.shape[0])
        for i in range(self.input_signal.shape[0]):
            peak_values[i] = np.max(np.abs(self.input_signal[i]))
        return peak_values

    def signal_absolute_mean_value(self):
        """Finds the absolute mean value of a signal.

        Returns:
          A float representing the absolute mean value of the signal.
        """

        # Calculate the absolute value of the audio signal.
        absolute_input_signal= np.absolute(self.input_signal)

        # Calculate the mean of the absolute audio signal.
        signal_abs_mean_value = np.mean(absolute_input_signal)

        return signal_abs_mean_value

    def signal_rms(self):
        """Calculates the RMS value of a signal.

        Returns:
          A NumPy array containing the RMS value of the audio signal.
        """

        sig_rms = np.sqrt(np.mean(np.square(self.input_signal)))
        return sig_rms

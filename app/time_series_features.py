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
        """
        Finds the maximum and minimum value of a signal.

        Returns:
          A tuple of two floats, representing the maximum and minimum value of the signal.
        """

        # Find the maximum and minimum value of the audio signal.
        signal_max_value = np.max(self.input_signal)
        signal_min_value = np.min(self.input_signal)

        return signal_max_value, signal_min_value

    def signal_peak_values(self):
        """
        Calculates the peak values of a signal.

        Returns:
          A NumPy array containing the peak values of the signal.
        """

        peak_values = np.zeros(self.input_signal.shape[0])
        for i in range(self.input_signal.shape[0]):
            peak_values[i] = np.max(np.abs(self.input_signal[i]))
        return peak_values

    def signal_absolute_mean_value(self):
        """
        Finds the absolute mean value of a signal.

        Returns:
          A float representing the absolute mean value of the signal.
        """

        # Calculate the absolute value of the audio signal.
        absolute_input_signal= np.absolute(self.input_signal)

        # Calculate the mean of the absolute audio signal.
        signal_abs_mean_value = np.mean(absolute_input_signal)

        return signal_abs_mean_value

    def signal_rms(self):
        """
        Calculates the RMS value of a signal.

        Returns:
          A NumPy array containing the RMS value of the signal.
        """

        sig_rms = np.sqrt(np.mean(np.square(self.input_signal)))
        return sig_rms

    def signal_standard_deviation(self):
        """
        Calculates the standard deviation of a signal.

        Returns:
          A NumPy array containing the standard deviation of the signal.
        """

        sig_standard_deviation = np.std(self.input_signal)
        return sig_standard_deviation

    def signal_standard_deviation_in_frequency_domain(self):
        """
        Calculates the standard deviation of a signal in the frequency domain.

        Returns:
          A NumPy array containing the standard deviation of the signal in the frequency domain.
        """

        stft = librosa.core.stft(self.input_signal, n_fft=self.frame_length, hop_length=self.hop_length)

        # Calculate the standard deviation of the magnitude spectrum

        sig_f_standard_deviation = np.std(np.abs(stft), axis=1)

        # Convert the standard deviation back to the time domain

        istft = librosa.core.istft(sig_f_standard_deviation, n_fft=self.frame_length, hop_length=self.hop_length)

        return istft

    def signal_variance(self):
        """
        Calculates the variance of a signal.

        Returns:
          A NumPy array containing the variance of the signal.
        """

        variance = np.var(self.input_signal, axis=0)
        return variance

    def signal_skewness(self):
        """
        Calculates the skewness of a signal.

        Returns:
          A NumPy array containing the skewness of the signal.
        """

        # Calculate the magnitude spectrum of the signal.
        magnitude_spectrum = librosa.core.stft(self.input_signal,
                                               n_fft=self.frame_length,
                                               hop_length=self.hop_length)
        magnitude_spectrum = np.abs(magnitude_spectrum)

        # Calculate the skewness of the magnitude spectrum.
        skewness = np.mean(magnitude_spectrum ** 3) / np.std(magnitude_spectrum) ** 3

        return skewness

    def signal_kurtosis(self):
        """
        Calculates the kurtosis of a signal.

        Returns:
          A NumPy array containing the kurtosis of the signal.
        """

        # Calculate the magnitude spectrum of the signal.
        magnitude_spectrum = librosa.core.stft(self.input_signal,
                                               n_fft=self.frame_length,
                                               hop_length=self.hop_length)
        magnitude_spectrum = np.abs(magnitude_spectrum)

        # Calculate the kurtosis of the magnitude spectrum.
        kurtosis = np.mean(magnitude_spectrum ** 4) / np.std(magnitude_spectrum) ** 4 - 3

        return kurtosis

    def signal_waveform_factor(self):
        """
        Calculates the waveform factor of a signal.

        Returns:
          float: The waveform factor.
        """

        average = np.mean(self.input_signal)
        peak = np.max(np.abs(self.input_signal))
        return peak / average

    def signal_crest_factor(self):
        """
        Calculates the crest factor of a signal.

        Returns:
          float: The crest factor.
        """

        peak = np.max(np.abs(self.input_signal))
        rms = np.sqrt(np.mean(self.input_signal ** 2))
        return peak / rms

    def signal_clearance_factor(self):
        """
        Calculates a simple clearance factor for a signal.

        Returns:
          float: The clearance factor.
        """

        # Calculate the average and peak amplitudes of the signal.
        average = np.mean(self.input_signal)
        peak = np.max(np.abs(self.input_signal))

        # Calculate the clearance factor.
        clearance_factor = (peak - average) / average

        return clearance_factor

    def signal_pulse_factor(self):
        """
        Calculates a simple pulse factor for a signal.

        Returns:
          float: The pulse factor.
        """

        # Calculate the average and peak amplitudes of the signal.
        average = np.mean(self.input_signal)
        peak = np.max(np.abs(self.input_signal))

        # Calculate the pulse factor.
        pulse_factor = (peak - average) / (average + peak)

        return pulse_factor

    def signal_fft(self):
        """
        Computes the fast Fourier transform (FFT) of a signal.

        Returns:
          A NumPy array containing the FFT of the signal.
        """

        fft_signal = np.fft.fft(self.input_signal)
        return fft_signal

    def signal_stft(self):
        """
        Computes the Short Time Fourier Transform (STFT) of a signal.

        Returns:
          A NumPy array containing the STFT of the signal.
        """

        # Compute the STFT.
        D = librosa.stft(self.input_signal, n_fft=self.frame_length, hop_length=self.hop_length)

        # Return the STFT.
        return D

    def signal_spectrogram(self):
        """
        Computes the spectrogram of a signal using the Librosa library.

        Returns:
          A NumPy array containing the spectrogram of the signal.
        """

        # Compute the spectrogram.
        S = librosa.core.stft(self.input_signal, n_fft=self.frame_length, hop_length=self.hop_length)

        # Compute the magnitude of the spectrogram.
        magnitude = np.abs(S)

        # Compute the spectrogram in decibels.
        spectrogram = 20 * np.log10(magnitude)

        # Return the spectrogram.
        return spectrogram

    def signal_mel_spectrogram(self, N_mels=64):
        """
        Calculates the Mel Spectral coefficients!!!

        :param N_mels: number of Mel bands to generate!

        :return: Mel transform matrix, Mel transform matrix in DB.
        """
        # Compute mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=self.input_signal, sr=self.sample_rate,
                                                         n_fft=self.frame_length, hop_length=self.hop_length,
                                                         n_mels=N_mels)

        # Convert power spectrogram to dB scale
        mel_S_dB = librosa.power_to_db(mel_spectrogram, ref=np.max)

        return mel_spectrogram, mel_S_dB

    def signal_wavelet_transform(self, wavelet_name='db4'):
        """
        Calculates the wavelet transform of a signal.

        Args:
          wavelet_name: The name of the wavelet to use.

        Returns:
          A NumPy array containing the wavelet coefficients.
        """

        # Calculate the wavelet coefficients.
        wave_coeffs = pywt.wavedec(self.input_signal, wavelet_name)

        # Return the wavelet coefficients.
        return wave_coeffs

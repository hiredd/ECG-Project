import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate
from scipy.io import loadmat
from scipy import signal, fftpack
plt.close("all")


def filter_ecg(ecg, fc1, fc2, fs):
    """
    Parameters:
    -----------
    ecg : array of floats
        ECG signal array.
    fc1 : float
        Lower cutoff frequency (Hz).
    fc2 : float
        Upper cutoff fbtw requency (Hz).
    fs : float
        Sampling frequency (Hz).

    Returns:
    --------
    filtered_ecg : array of floats
        Filtered ECG signal.
    """

    nyquist_frequency = fs / 2
    lowcut = fc1 / nyquist_frequency
    highcut = fc2 / nyquist_frequency
    b, a = signal.butter(8, [lowcut, highcut], btype='band')
    # Numerator (b) and denominator (a) polynomials of the IIR filter.
    # Butterworth filter has as flat a frequency response as possible in passband.
    filtered_ecg = signal.filtfilt(b, a, ecg)
    # Provides the filtered output with the same shape as x.

    return filtered_ecg


def find_r_peaks(filtered_ecg, fs):
    """
    Parameters:
    -----------
    filtered_ecg : array of floats
        Filtered ECG signal array.
    fs : float
        Sampling frequency (Hz).

    Returns:
    --------
    r_peaks_idx : array of ints
    Array containing indices (positions) of R-peaks.
    """

    ecg_length = len(filtered_ecg)
    tolerantion = int(0.1*fs)

    temporary_peaks = [i + np.argmax(filtered_ecg[i:i + fs]) for i in range(ecg_length)]

    indices_buffer = set()
    for i in temporary_peaks:
        before = i - tolerantion
        if before < 0:
            continue
        after = i + tolerantion
        if after > (ecg_length+fs):
            break

        peak_index = before + np.argmax(filtered_ecg[before:after])
        indices_buffer.add(peak_index)

    r_peaks_idx = sorted(list(indices_buffer))

    return r_peaks_idx


def calc_freq_content(ecg, f_max, fs):
    """
    Parameters:
    -----------
    ecg : array of floats
        ECG signal array.
    f_max : float
        Upper bound of frequency band (Hz).
    fs : float
        Sampling frequency (Hz).

    Returns:
    --------
    freq_content : float
        Percentage of energy in band (0, f_max Hz).
    """
    x_acis = int((len(ecg))/fs)
    FFT = fftpack.fft(ecg, n=x_acis)
    FFT = abs(FFT[0:int(x_acis/2)])

    E1 = (FFT[0:int(f_max)])**2
    E1 = integrate.trapz(E1)
    E2 = FFT**2
    E2 = integrate.trapz(E2)

    freq_content = (E1/E2)*100

    return freq_content

""" Load data """
task_data = loadmat("data.mat")
ecg = task_data["data"]
fs = task_data["fs"]

""" Run your code """

ecg = ecg[0]
fs = fs[0][0]

filtered_ecg = filter_ecg(ecg=ecg, fc1=8.0, fc2=40.0, fs=fs)
r_peaks_idx = find_r_peaks(filtered_ecg=filtered_ecg, fs=fs)
freq_content = calc_freq_content(ecg=ecg, f_max=15.0, fs=fs)

""" Display results """
plt.figure(1)
plt.xlabel("Sample number")
plt.ylabel("Value")
plt.plot(ecg, label="RAW ECG")
plt.plot(filtered_ecg, label="Filtered ECG")
plt.plot(r_peaks_idx, filtered_ecg[r_peaks_idx], "o", label="R-peaks")
plt.legend(bbox_to_anchor=(1, 1.02), loc=4, borderaxespad=0)
print("Percentage of energy in band (0,15Hz): %.2f" % freq_content)
plt.show()

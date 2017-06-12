from scipy.io import loadmat
import matplotlib.pyplot as plt
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
        Upper cutoff frequency (Hz).
    fs : float
        Sampling frequency (Hz).
        
    Returns:
    --------
    filtered_ecg : array of floats
        Filtered ECG signal.
    """

    # Type your code here...

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
    # Type your code here...

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
    # Type your code here...

    return freq_content


""" Load data """
task_data = loadmat("data.mat")
ecg = task_data["data"]
fs = task_data["fs"]

""" Run your code """
filtered_ecg = filter_ecg(ecg=ecg, fc1=8.0, fc2=40.0, fs=fs)
r_peaks_idx = find_r_peaks(filtered_ecg=filtered_ecg, fs=fs)
freq_content = calc_freq_content(ecg=ecg, f_max=15.0, fs=fs)

""" Display results """
plt.figure()
plt.plot(filtered_ecg)
plt.plot(r_peaks_idx, filtered_ecg[r_peaks_idx],"o")
print("Percentage of energy in band (0,15Hz): %.2f" % freq_content)

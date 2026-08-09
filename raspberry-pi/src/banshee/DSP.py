import numpy as np

def cross_correlation(signal, kernel, fft_len=None):
    if fft_len is None:
        fft_len = signal + kernel - 1
    return np.fft.irfft( np.fft.rfft(signal, fft_len) * np.fft.rfft(kernel[::-1], fft_len) )

def signal_delay(cross_corrl_signal, n_samps, samp_rate):
    # return n_samps - np.argmax(cross_corrl_signal[:n_samps]) / samp_rate # [s]
    print("argmax", np.argmax(cross_corrl_signal))
    print("lag [samples]", n_samps - np.argmax(cross_corrl_signal))
    return (n_samps - np.argmax(cross_corrl_signal)) / samp_rate # [s]

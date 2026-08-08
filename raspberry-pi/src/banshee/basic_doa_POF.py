import numpy as np
from matplotlib import pyplot as plt

from mock_audio import get_mic_audio

n_mics = 2
time_win = 1 # [s]
samp_rate = 1000 # [Hz]
n_samps = samp_rate // time_win

def main():
    audio_mat_shape = (n_mics, n_samps)
    mic_audio = np.zeros(audio_mat_shape)
    fft_len = n_samps*2-1
    # mic_phase_diffs = np.zeros((n_mics-1, n_samps))
    # mic_phase_diffs = np.zeros((n_mics-1, fft_len))
    mic_cross_corrls = np.zeros((n_mics-1, fft_len-1))
    mic_delays = np.zeros(n_mics-1)

    for idx_mic in range(n_mics):
        mic_audio[idx_mic] = get_mic_audio(n_samps, idx_mic)

    fig, ax = plt.subplots(2,1)
    ax[0].plot(mic_audio[0])
    ax[1].plot(mic_audio[1])

    for idx_mic in range(n_mics-1):
        # mic_phase_diffs[idx_mic] = np.correlate(mic_audio[0], mic_audio[idx_mic+1], "full")
        # mic_phase_diffs[idx_mic] = np.abs(np.convolve(mic_audio[0], np.flip(mic_audio[idx_mic+1]), 'full'))
        # mic_phase_diffs[idx_mic] = np.abs(np.fft.irfft( np.fft.rfft(mic_audio[0]) * np.fft.rfft(mic_audio[idx_mic+1][::-1]) ))
        mic_cross_corrls[idx_mic] = np.fft.irfft( np.fft.rfft(mic_audio[0], fft_len) * np.fft.rfft(mic_audio[idx_mic+1][::-1], fft_len) )

    # lags = np.arange(-n_samps + 1, n_samps)
    lags = np.arange(-n_samps + 1, n_samps - 1)
    fig, ax = plt.subplots()
    ax.plot(lags, mic_cross_corrls[0])

    for idx_mic in range(n_mics-1):
        mic_delays[idx_mic] = np.abs(n_samps - np.argmax(mic_cross_corrls[idx_mic][:n_samps])) / samp_rate # [s]
    # print(mic_delays)

    plt.show()

if __name__ == "__main__":
    main()

import numpy as np
from matplotlib import pyplot as plt

from mock_audio import get_mic_audio
from DSP import cross_correlation, signal_delay

n_mics = 2
time_win = 1 # [s]
samp_rate = 1000 # [Hz]
n_samps = samp_rate * time_win + 1
speed_sound = 340.29 # [m/s] @ 20 C sea-level
mic_phys_dist = 340.29 # [m]
max_mic_delay = mic_phys_dist / speed_sound # [s] maximum theoretical delay between mics (sound travelling parallel to plane formed by mic pair)

def main():
    audio_mat_shape = (n_mics, n_samps)
    mic_audio = np.zeros(audio_mat_shape)
    fft_len = n_samps*2-1
    mic_cross_corrls = np.zeros((n_mics-1, fft_len-1))
    mic_delays = np.zeros(n_mics-1)
    mic_plane_bearing_est = np.zeros(n_mics-1)

    for idx_mic in range(n_mics):
        mic_audio[idx_mic] = get_mic_audio(n_samps, time_win, idx_mic)

    fig, ax = plt.subplots(2,1)
    ax[0].plot(mic_audio[0])
    ax[1].plot(mic_audio[1])

    for idx_mic in range(n_mics-1):
        asdf = cross_correlation(mic_audio[0], mic_audio[idx_mic+1], fft_len)
        mic_cross_corrls[idx_mic] = asdf

    lags = np.arange(-n_samps + 1, n_samps - 1)
    fig, ax = plt.subplots()
    ax.plot(lags, mic_cross_corrls[0])

    for idx_mic in range(n_mics-1):
        mic_delays[idx_mic] = signal_delay(mic_cross_corrls[idx_mic], n_samps, samp_rate)
    print(mic_delays)

    for idx_mic in range(n_mics-1):
        delay_norm = mic_delays[idx_mic] / max_mic_delay
        mic_plane_bearing_est[idx_mic] = 90 + delay_norm*90 # [deg]

    plt.show()

if __name__ == "__main__":
    main()

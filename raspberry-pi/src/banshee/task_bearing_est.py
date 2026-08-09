import numpy as np

from DSP import cross_correlation, signal_delay

n_mics = 4
mic_plane_offset = 360 // n_mics
# time_win = 2 # [s]
# samp_rate = 1000 # [Hz]
# n_samps = samp_rate * time_win + 1
# fft_len = n_samps*2-1
speed_sound = 340.29 # [m/s] @ 20 C sea-level
mic_phys_dist = 340.29 # [m]
max_mic_delay = mic_phys_dist / speed_sound # [s] maximum theoretical delay between mics (sound travelling parallel to plane formed by mic pair)

def loop_bearing_est(mic_audio, n_samps, samp_rate, time_win):
    fft_len = n_samps*2-1
    audio_mat_shape = (n_mics, n_samps)
    mic_audio = np.zeros(audio_mat_shape)
    # mic_cross_corrls = np.zeros((n_mics-1, fft_len-1))
    mic_cross_corrls_frt_bk = np.zeros((fft_len-1,))
    mic_cross_corrls_l_r = np.zeros((fft_len-1,))
    # mic_delays = np.zeros(n_mics-1)
    mic_delay_frt_bk = 0
    mic_delay_l_r = 0
    mic_plane_bearing_est = np.zeros(n_mics-1)

    # Sample lags
    mic_cross_corrls_frt_bk = cross_correlation(mic_audio[0], mic_audio[1], fft_len)
    mic_cross_corrls_l_r = cross_correlation(mic_audio[2], mic_audio[3], fft_len)

    # Time delays
    mic_delay_frt_bk = signal_delay(mic_cross_corrls_frt_bk, n_samps, samp_rate)
    mic_delay_l_r = signal_delay(mic_cross_corrls_l_r, n_samps, samp_rate)

    # Bearing est
        # Plane relative bearings
    delay_norm_frt_bk = mic_delay_frt_bk / max_mic_delay
    mic_plane_bearing_est_frt_bk = np.clip( delay_norm_frt_bk*180, a_min=0, a_max=180 ) # [deg]
    delay_norm_l_r = mic_delay_l_r / max_mic_delay
    mic_plane_bearing_est_l_r = np.clip( delay_norm_l_r*180, a_min=0, a_max=180 ) # [deg]
    mic_plane_bearing_est = np.array([mic_plane_bearing_est_frt_bk, mic_plane_bearing_est_l_r])
        # Absolute bearing of possible directions
    bearing_est_permutations = np.zeros((n_mics,))
    for perm_idx in range(n_mics//2):
        bearing_est_permutations[perm_idx*2] = 360 - mic_plane_bearing_est[perm_idx] + mic_plane_offset*perm_idx
        bearing_est_permutations[perm_idx*2+1] = mic_plane_bearing_est[perm_idx] + mic_plane_offset*perm_idx
    #     # Distance matrix of possibles
    # bearings_dist_mat = np.matmul(bearing_est_permutations[:n_mics//2], bearing_est_permutations)
    bearing_est = np.quantile(bearing_est_permutations, 0.25)
    print(bearing_est)
    return bearing_est

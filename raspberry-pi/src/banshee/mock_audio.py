import numpy as np

def get_mic_audio(num_samples, time_win, idx_mic):
    t = np.linspace(0, time_win, num_samples)
    # return np.sin(2*np.pi * t + idx_mic*np.pi/16) + np.random.uniform(-1,1,num_samples)
    return np.sin(2*np.pi * 1 * t - (idx_mic*np.pi/2))

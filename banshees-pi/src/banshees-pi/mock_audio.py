import numpy as np

def get_mic_audio(num_samples, idx_mic):
    t = np.linspace(0, 1, num_samples)
    # return np.sin(2*np.pi * (t + idx_mic*num_samples*0.6)) + np.random.uniform(-1,1,num_samples)
    return np.sin(2*np.pi * 1 * t - (idx_mic*np.pi/16))

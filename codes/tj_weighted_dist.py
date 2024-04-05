#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import correlate, find_peaks
import sounddevice as sd
import wave
# matplotlib qt


# # Microphone Parameters
# 1. Mic Gain: 80
# 2. Recording Mode: Ambisonics FuMa
# 3. Recording Format (Sampling Rate): 48kHz
# 4. Recording Format (Bit depth): 24-bit

# In[2]:


# Step 1: Define the parameters

# Reference: Patricio et. al in [14] E. Patricio, A. Ruminski, A. Kuklasinski, 
# L. Januszkiewicz, and T. Zernicki, "Toward Six Degrees of Freedom Audio 
# Recording and Playback Using Multiple Ambisonics Sound Fields," Paper 10141, 
# (2019 March.)

# Number of microphones
M = 3

# Order of Ambisonics
# We are using first order ambisonics in FuMa Format
# Note: WAV Format: W,X,Y,Z
P = 4

# Placing the ambisonic microphones in a triangle, this is the length of the 
# triangle (in centimeters)
triangle_side = 128

# Constants for the attenuation function
volume_threshold = 0.9
volume_range = 0.9
hoa_threshold = 0.9
hoa_range = 1.3

# Defining the indices of the channel
w_channel = 0
x_channel = 1
y_channel = 2
z_channel = 3

# Defining the microphone positions as a global parameter
mic_positions = np.zeros((3, 2))

mic_positions[0, :] = [0, 0]
mic_positions[1, :] = [triangle_side, 0]
mic_positions[2, :] = [triangle_side / 2, (triangle_side * np.sqrt(3)) / 2]


# # Calibrating the microphones

# In[3]:


# Getting the 440Hz Sine Tone
fs, mic_1_calib = wavfile.read('../data/calibration_recordings/02_28/mic1_a440_2.WAV')
_, mic_2_calib = wavfile.read('../data/calibration_recordings/02_28/mic2_a440_2.WAV')
_, mic_3_calib = wavfile.read('../data/calibration_recordings/02_28/mic3_a440_2.WAV')
_, gt_calib = wavfile.read('../data/calibration_recordings/02_28/gt_a440_2.WAV')

# Note: Ground truth recording for 02_28 was incorrect gain level

# Finding the sampling rate
print(f'Sampling rate: {fs} ')

start = 1000
end = 3500

# Getting the omnidirectional channels
mic1_w = mic_1_calib[:, 0].astype(float)
mic2_w = mic_2_calib[:, 0].astype(float)
mic3_w = mic_3_calib[:, 0].astype(float)
gt_w = gt_calib[:, 0].astype(float)

# Shorten the w components
mic1_w_shortened = mic1_w[start:end]
mic2_w_shortened = mic2_w[start:end]
mic3_w_shortened = mic3_w[start:end]
gt_w_shortened = gt_w[start:end]

# Plotting the tone
time_shortened = np.arange(1, mic1_w_shortened.shape[0] + 1)

# Plotting the graph
plt.plot(time_shortened, mic1_w_shortened, color='blue')
plt.title('Mic 1 Sine Wave')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()

plt.plot(time_shortened, mic2_w_shortened, color='red')
plt.title('Mic 2 Sine Wave')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()

plt.plot(time_shortened, mic3_w_shortened, color='green')
plt.title('Mic 3 Sine Wave')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()

plt.plot(time_shortened, gt_w_shortened, color='orange')
plt.title('Ground Truth Sine Wave')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()


# # Analyzing the outputs from microphones
# 
# 1. Place the microphones 2m away from the sound source. Note that all microphones should be facing the same direction
# 2. Play the 440Hz sine wave with a flat speaker, perform a 10 second recording with the microphones.
# 3. Load all four microphone outputs here. 
# 4. Find some equation to normalize the sound to the same amplitude. 

# In[4]:


from scipy.fft import fft
def plot_frequency_response(signal, fs, title):
    n = len(signal)
    freq = np.fft.fftfreq(n, d=1/fs)
    spectrum = fft(signal)

    # Calibrating the y-values to be within a range of 0-1000
    plt.plot(freq[:n//2], np.abs(spectrum[:n//2]))
    plt.title(title)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.show()

plot_frequency_response(mic1_w_shortened, fs, 'Mic 1 Frequency Response')
plot_frequency_response(mic2_w_shortened, fs, 'Mic 2 Frequency Response')
plot_frequency_response(mic3_w_shortened, fs, 'Mic 3 Frequency Response')
plot_frequency_response(gt_w_shortened, fs, 'Ground Truth Frequency Response')

# Normalizing each of the microphones

def normalize_mic(signal):
    return (signal - np.mean(signal)) / np.std(signal)

mic1_w_normalized = normalize_mic(mic1_w_shortened)
mic2_w_normalized = normalize_mic(mic2_w_shortened)
mic3_w_normalized = normalize_mic(mic3_w_shortened)
gt_w_normalized = normalize_mic(gt_w_shortened)

# Plotting the normalized signals

def plot_signal(time, signal, title, color='blue'):
    plt.plot(time_shortened, mic1_w_shortened, color)
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.show()

plot_signal(time_shortened, mic1_w_normalized, 'Mic 1 Sine Wave', color='blue')
plot_signal(time_shortened, mic2_w_normalized, 'Mic 2 Sine Wave', color='red')
plot_signal(time_shortened, mic3_w_normalized, 'Mic 3 Sine Wave', color='green')
plot_signal(time_shortened, gt_w_normalized, 'Ground Truth Sine Wave', color='orange')

# Function to plot four different waves
def plot_waves(time_axis, signal, title, xlabel='Time', ylabel='Amplitude'):
    # Extracting the channels
    w, x, y, z = extract_channels(signal)

    # Plotting the W componment
    plt.subplot(2, 2, 1)
    plt.plot(time_axis, w, color='blue')
    plt.title(f'{title} W component')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Plotting the X component
    plt.subplot(2, 2, 2)
    plt.plot(time_axis, x, color='green')
    plt.title(f'{title} X component')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Plotting the Y component
    plt.subplot(2, 2, 3)
    plt.plot(time_axis, y, color='red')
    plt.title(f'{title} Y component')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Plotting the Z component
    plt.subplot(2, 2, 4)
    plt.plot(time_axis, z, color='orange')
    plt.title(f'{title} Z component')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.tight_layout()
    plt.show()


# # Functions for wav file processing

# In[5]:


from pydub import AudioSegment

def get_num_channels(wav_file_path):
    audio = AudioSegment.from_file(file=wav_file_path, format="wav")
    num_channels = audio.channels
    return num_channels


# # Functions to calculate the attenuation of different microphones. 

# In[6]:


def attenuation_and_rebalancing(d_m, t_l=volume_threshold, t_k=hoa_range, s_k_0=1.0, s_k_p_neg=-1.0, s_l_neg=-1.0):
    """
    Distance-dependent attenuation and component re-balancing function.

    Parameters:
    - d_m: distance to the m-th microphone
    - t_l: attenuation threshold
    - t_k: re-balancing threshold
    - s_k_0: slope for the 0th order component
    - s_k_p_neg: slope for higher-order components (p>0)
    - s_l_neg: slope for overall gain

    Returns:
    - Attenuation coefficient for the given distance
    """
    # Equation (3)
    def l(d_m):
        # Piecewise function where: 
        # l(d_m) = 0 if d_m <= t_l
        # l(d_m) = s_l(d_m - t_l) otherwise
        return np.where(d_m <= t_l, 0, s_l(d_m - t_l))

    # Equation (4)
    def k_p(d_m):
        # Piecewise function where: 
        # k_p(d_m) = 0 if d_m <= t_k
        # k_p(d_m) = s_k_p(d_m - t_k) otherwise
        return np.where(d_m <= t_k, 0, s_k_p(d_m - t_k))

    # Component in equation (3)
    def s_l(delta_d):
        # Adjust this function based on the specific behavior of s_l
        # For example, you can use a linear function: return s_l_neg * delta_d
        return s_l_neg * delta_d

    # Component in equation (4)
    def s_k_p(delta_d):
        # Adjust this function based on the specific behavior of s_k_p
        # For p=0, the slope is positive; for p>0, the slope is negative
        return np.where(delta_d <= 0, s_k_0, s_k_p_neg * delta_d)

    # Equation (2)
    return 10 ** ((l(d_m) + k_p(d_m)) / 20.0)

# Function to compute the distance between the interpolation points and the microphone
def compute_distance(interp_point, mic_number, triangle_side):
    
    # Euclidean distance between the interpolation point and the microphone

    distance = np.linalg.norm(interp_point - mic_positions[mic_number, :])
    return distance

# Function to extract the w, x, y, z channel signals
def extract_channels(mic_signal):
    # W Channel (Omni directional)
    w_signal = mic_signal[:, w_channel].astype(float)
    # X Channel
    x_signal = mic_signal[:, x_channel].astype(float)
    # Y Channel
    y_signal = mic_signal[:, w_channel].astype(float)
    # Z Channel
    z_signal = mic_signal[:, w_channel].astype(float)

    return w_signal, x_signal, y_signal, z_signal

# Function to get back the original signals
def combine_channels(w_signal, x_signal, y_signal, z_signal):
    assert(len(w_signal) == len(x_signal) == len(y_signal) == len(z_signal))
    # Create a microphone signal
    mic_signal = np.zeros((len(w_signal), P))

    # Combine them together
    mic_signal[:, w_channel] = w_signal
    mic_signal[:, x_channel] = x_signal
    mic_signal[:, y_channel] = y_signal
    mic_signal[:, z_channel] = z_signal

    return mic_signal

# Getting the minimum length of the signals
def get_min_length(mic1_signal, mic2_signal, mic3_signal):
    return min(len(mic1_signal), len(mic2_signal), len(mic3_signal))



# In[7]:


# Function to plot four different waves
def plot_waves(time_axis, signal, title, xlabel='Time', ylabel='Amplitude'):
    # Extracting the channels
    w, x, y, z = extract_channels(signal)

    # Plotting the W componment
    plt.subplot(2, 2, 1)
    plt.plot(time_axis, w, color='blue')
    plt.title(f'{title} W component')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Plotting the X component
    plt.subplot(2, 2, 2)
    plt.plot(time_axis, x, color='green')
    plt.title(f'{title} X component')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Plotting the Y component
    plt.subplot(2, 2, 3)
    plt.plot(time_axis, y, color='red')
    plt.title(f'{title} Y component')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Plotting the Z component
    plt.subplot(2, 2, 4)
    plt.plot(time_axis, z, color='orange')
    plt.title(f'{title} Z component')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.tight_layout()
    plt.show()


# # Finding the time lag among the three different microphone recordings

# In[8]:


import numpy as np
import scipy.signal
from scipy.io import wavfile

fs, mic1_signal = wavfile.read('../data/02_14/mic1.WAV')
_, mic2_signal = wavfile.read('../data/02_14/mic2.WAV')
_, mic3_signal = wavfile.read('../data/02_14/mic3.WAV')

# Extracting the data from each channel
mic1_w_signal, mic1_x_signal, mic1_y_signal, mic1_z_signal = extract_channels(mic1_signal)
mic2_w_signal, mic2_x_signal, mic2_y_signal, mic2_z_signal = extract_channels(mic2_signal)
mic3_w_signal, mic3_x_signal, mic3_y_signal, mic3_z_signal = extract_channels(mic3_signal)

# Making the microphone signals all the same length as the lowest one
min_w_length = get_min_length(mic1_w_signal, mic2_w_signal, mic3_w_signal)
min_x_length = get_min_length(mic1_x_signal, mic2_x_signal, mic3_x_signal)
min_y_length = get_min_length(mic1_y_signal, mic2_y_signal, mic3_y_signal)
min_z_length = get_min_length(mic1_z_signal, mic2_z_signal, mic3_z_signal)
assert(min_w_length == min_x_length == min_y_length == min_z_length)

start = 200000 # Find some way to change this, it's a bit arbitrary now. 
min_length = min_w_length # Once we confirmed that the channel lengths are the same

# Making all the signals the minimum length
# Mic 1
mic1_w_signal = mic1_w_signal[start:min_length]
mic1_x_signal = mic1_x_signal[start:min_length]
mic1_y_signal = mic1_y_signal[start:min_length]
mic1_z_signal = mic1_z_signal[start:min_length]

# Mic 2
mic2_w_signal = mic2_w_signal[start:min_length]
mic2_x_signal = mic2_x_signal[start:min_length]
mic2_y_signal = mic2_y_signal[start:min_length]
mic2_z_signal = mic2_z_signal[start:min_length]

# Mic 3
mic3_w_signal = mic3_w_signal[start:min_length]
mic3_x_signal = mic3_x_signal[start:min_length]
mic3_y_signal = mic3_y_signal[start:min_length]
mic3_z_signal = mic3_z_signal[start:min_length]

# Re-organizing them in terms of their directional components
w_matrix = np.array([mic1_w_signal, mic2_w_signal, mic3_w_signal])
x_matrix = np.array([mic1_x_signal, mic2_x_signal, mic3_x_signal])
y_matrix = np.array([mic1_y_signal, mic2_y_signal, mic3_y_signal])
z_matrix = np.array([mic1_z_signal, mic2_z_signal, mic3_z_signal])

# Regenerating the wav files from the signals
# Perform the Fourier transform of the omnidirectional signals
f_mic1 = np.fft.fft(mic1_w_signal)
f_mic2 = np.fft.fft(mic2_w_signal)
f_mic3 = np.fft.fft(mic3_w_signal)

# # Calculate cross-correlation in the frequency domain
# # np.conj: Finds the complex conjugate of the fourier transform

correlation12 = np.fft.ifft(f_mic1 * np.conj(f_mic2))
correlation13 = np.fft.ifft(f_mic1 * np.conj(f_mic3))
correlation23 = np.fft.ifft(f_mic2 * np.conj(f_mic3))

print(f'Correlation12: {correlation12}')
print(f'Correlation13: {correlation13}')
print(f'Correlation23: {correlation23}')

# Find the lag with maximum correlation
lag12 = np.argmax(np.abs(correlation12))
lag13 = np.argmax(np.abs(correlation13))
lag23 = np.argmax(np.abs(correlation23))

# Adjust for potential wrap-around effect
N = len(mic1_w_signal)
lag12 = lag12 if lag12 <= N // 2 else lag12 - N
lag13 = lag13 if lag13 <= N // 2 else lag13 - N
lag23 = lag23 if lag23 <= N // 2 else lag23 - N

# Calculate time delays
delay12 = lag12 / fs
delay13 = lag13 / fs
delay23 = lag23 / fs

print(f'Time delay between mic1 and mic2: {delay12:.5f} seconds')
print(f'Time delay between mic1 and mic3: {delay13:.5f} seconds')
print(f'Time delay between mic2 and mic3: {delay23:.5f} seconds')

lag = np.array([lag13, lag23, 0])


# In[9]:


# Convert time delays to sample indices
delay12_samples = int(round(delay12 * fs))
delay13_samples = int(round(delay13 * fs))
delay23_samples = delay12_samples - delay13_samples  # Derived from delay12 and delay13

# Find the maximum delay to align signals
max_delay_samples = max(0, delay12_samples, delay13_samples, -delay23_samples)

# Initialize the array to store aligned signals
aligned_w_signals = []
aligned_x_signals = []
aligned_y_signals = []
aligned_z_signals = []

# Align each signal based on the calculated delays
for i in range(1, 4):
    if i == 0:  # mic 1
        delay = max_delay_samples - 0
    elif i == 1:  # mic 2
        delay = max_delay_samples - delay12_samples
    else:  # mic 3
        delay = max_delay_samples - delay13_samples

    # 

    # Apply the delay (pad with zeros at the beginning and trim the end)
    w_signal = eval(f'mic{i}_w_signal')
    x_signal = eval(f'mic{i}_x_signal')
    y_signal = eval(f'mic{i}_y_signal')
    z_signal = eval(f'mic{i}_z_signal')

    padded_w_signal = np.pad(w_signal, (delay, 0), mode='constant', constant_values=0)
    padded_x_signal = np.pad(x_signal, (delay, 0), mode='constant', constant_values=0)
    padded_y_signal = np.pad(y_signal, (delay, 0), mode='constant', constant_values=0)
    padded_z_signal = np.pad(z_signal, (delay, 0), mode='constant', constant_values=0)

    # Adding to the array of aligned signals
    aligned_w_signals.append(padded_w_signal[:len(w_signal)])
    aligned_x_signals.append(padded_x_signal[:len(x_signal)])
    aligned_y_signals.append(padded_y_signal[:len(y_signal)])
    aligned_z_signals.append(padded_z_signal[:len(z_signal)])

# Getting the aligned components
mic1_w_aligned = aligned_w_signals[0]
mic2_w_aligned = aligned_w_signals[1]
mic3_w_aligned = aligned_w_signals[2]

mic1_x_aligned = aligned_x_signals[0]
mic2_x_aligned = aligned_x_signals[1]
mic3_x_aligned = aligned_x_signals[2]

mic1_y_aligned = aligned_y_signals[0]
mic2_y_aligned = aligned_y_signals[1]
mic3_y_aligned = aligned_y_signals[2]

mic1_z_aligned = aligned_z_signals[0]
mic2_z_aligned = aligned_z_signals[1]
mic3_z_aligned = aligned_z_signals[2]

mic_1_time = np.arange(1, mic1_w_aligned.shape[0] + 1)
mic_2_time = np.arange(1, mic2_w_aligned.shape[0] + 1)
mic_3_time = np.arange(1, mic3_w_aligned.shape[0] + 1)
assert(len(mic_1_time) == len(mic_2_time) == len(mic_3_time))

# Create y_m_p, which is the aligned signals
# Each item is the w, x, y, and z microphones
y_m_p = np.array([
    [mic1_w_aligned, mic2_w_aligned, mic3_w_aligned],
    [mic1_x_aligned, mic2_x_aligned, mic3_x_aligned],
    [mic1_y_aligned, mic2_y_aligned, mic3_y_aligned],
    [mic1_z_aligned, mic2_z_aligned, mic3_z_aligned]
])

# Plotting each of the signals
def plot_signal(time, signal, title, color='blue'):
    plt.plot(time, signal, color)
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.show()

# Plotting the original signals
# plot_signal(mic_1_time, mic1_w_signal, 'Microphone 1 signal (Original) ', color='blue')
# plot_signal(mic_2_time, mic2_w_signal, 'Microphone 2 signal (Original)', color='red')
# plot_signal(mic_3_time, mic3_w_signal, 'Microphone 3 signal (Original)', color='green')

# # Plotting the aligned signals
# plot_signal(mic_1_time, mic1_w_aligned, 'Microphone 1 signal (Aligned) ', color='blue')
# plot_signal(mic_2_time, mic2_w_aligned, 'Microphone 2 signal (Aligned)', color='red')
# plot_signal(mic_3_time, mic3_w_aligned, 'Microphone 3 signal (Aligned)', color='green')



# In[10]:


# Interpolation point coordinates

# Centroid
interp_point = np.array([triangle_side / 2, (triangle_side * np.sqrt(3)) / 4])  # Ground truth point
print(f"Interpolated Point: {interp_point}")

# Compute the distance for all samples at once (This is the d_m function)
distances = np.array([compute_distance(interp_point, m, triangle_side) for m in range(M)])

# Compute the attenuation coefficients for each microphone using broadcasting (This calculates the a_p(d_m))
a_p_values = attenuation_and_rebalancing(distances[:, np.newaxis])
a_p_values_reshaped = a_p_values.T[:, :, np.newaxis]

# Getting x_p 
x_p = np.sum(a_p_values_reshaped * y_m_p, axis=1)
x_p = x_p.T
x_p = x_p/1000.0
print(x_p)


# Plot
time_axis = np.arange(1, x_p.shape[0] + 1)
wavfile.write('output_centroid.wav', fs, x_p)

# Plotting the interpolated signal

plot_waves(time_axis, x_p, 'Interpolated signal')

# # Plotting the ground truth signal
_, gt_signal = wavfile.read('../data/02_14/gt.WAV')
gt_time = np.arange(1, gt_signal.shape[0] + 1)
plot_waves(gt_time, gt_signal, 'Ground Truth Signal')


# In[11]:


# Performing cross correlation
from scipy.signal import correlate

# First, we want to normalize it on the same amplitude scale
# x_p_normalized = (x_p - np.mean(x_p)) / np.std(x_p)
# gt_signal_normalized = (gt_signal - np.mean(gt_signal)) / np.std(gt_signal)

# Trying once without normalizing them
x_p_normalized = x_p
gt_signal_normalized = gt_signal

# Plot the normalized graphs
# plot_waves(time_axis, x_p_normalized, 'Interpolated Signal (Normalized)')

# plot_waves(gt_time, gt_signal_normalized, 'Ground Truth (Normalized)')

# Calculate cross-correlation
correlation_result = correlate(x_p_normalized, gt_signal_normalized, mode='full')

# Calculate normalized cross-correlation coefficient
norm_cross_corr_coeff = correlation_result / (len(x_p) * np.std(x_p) * np.std(gt_signal))

# Plot the similarity graph
plt.plot(norm_cross_corr_coeff)
plt.title('Normalized Cross-correlation Coefficient')
plt.xlabel('Lag')
plt.ylabel('Similarity')
plt.show()


# # Verification of maximum correlation
# Note: This is hardcoded

# In[12]:


# 1. Array of points of y-values
# points = np.arange(30, 81, 1)
points = [47.5, 50, 52.5, 55, 57.5, 60, 62.5]
num_points = len(points)
signals = []

# Array of cross correlation values
cross_corr_values = []

# Getting the values of cross_correlations
for i in range(num_points):
    y_point = points[i]
    interp_point = np.array([triangle_side / 2, y_point])
    distances = np.array([compute_distance(interp_point, m, triangle_side) for m in range(M)])
    a_p_values = attenuation_and_rebalancing(distances[:, np.newaxis])
    a_p_values_reshaped = a_p_values.T[:, :, np.newaxis]
    x_p = np.sum(a_p_values_reshaped * y_m_p, axis=1)
    x_p = x_p.T
    x_p = x_p/1000.0

    signals.append(x_p)

# Cross correlating the signals
i = 0
for signal in signals:
    print(f"Correlating: {i}")
    cross_corr = correlate(gt_signal_normalized, signal, mode='same')
    print(f"Successfully correlated: {i}")
    cross_corr_values.append(cross_corr)
    i += 1
    print(f"Correlating current y-point: {points[i]}")

# Identifying time shifts
time_shifts = [np.argmax(cross_corr) - (len(gt_signal_normalized) - 1) for cross_corr in cross_corr_values]

# Quantifying similarities
max_corr_values = [np.max(cross_corr) for cross_corr in cross_corr_values]
print(f"Max correlation values: {max_corr_values}")

most_similar_index = np.argmax(max_corr_values)
most_similar_signal = signals[most_similar_index]

# We expect it to say 55
print(f"The most signal is the one with y value: {points[most_similar_index]}")



# In[ ]:


# Normalize the max_corr_values:

# Plotting the graph
plt.plot(points, max_corr_values)
lower_bound = np.min(max_corr_values) - 0.0000000001
upper_bound = np.max(max_corr_values) + 0.0000000001
plt.ylim(lower_bound, upper_bound)
plt.title("Cross Correlation Graph")
plt.xlabel("Points of y-axis")
plt.ylabel("Similarity")
plt.show()


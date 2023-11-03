import pyaudio
import sys
import wave
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.signal import butter

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "output.wav"

p = pyaudio.PyAudio()



while True:
    input('type anything')
    print("* recording")
    stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)
    
    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save audio to a WAV file
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    # Convert frames to a NumPy array
    audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)

    # Perform Fast Fourier Transform (FFT)
    fft_result = np.fft.fft(audio_data)
    frequencies = np.fft.fftfreq(len(fft_result), 1 / RATE)  # Frequencies in Hz
    

    # Guitar string frequencies are 82 Hz, 110 Hz, 147 Hz, 196 Hz, 247 Hz, and 330 Hz

    freq = frequencies[:len(frequencies)//2]
    amp =  np.abs(fft_result)[:len(fft_result)//2]
    smoothed = butter(amp[0],0.77)
    peaks = find_peaks(amp,prominence=1000000)


    # Plot the frequency spectrum
    plt.figure(figsize=(10, 4))
    plt.plot(freq,amp)
    plt.plot(freq,smoothed)
    plt.plot(freq[peaks[0]],amp[peaks[0]],'rx')
    plt.ylim(0, 20000000)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title('Frequency Spectrum')
    plt.show()

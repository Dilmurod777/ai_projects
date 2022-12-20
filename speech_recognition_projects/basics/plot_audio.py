import wave
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    obj = wave.open("files/output.wav", "rb")

    frame_rate = obj.getframerate()
    n_frames = obj.getnframes()
    signal_wave = obj.readframes(-1)

    obj.close()

    duration = n_frames / frame_rate
    signal_array = np.frombuffer(signal_wave, dtype=np.int16)
    times = np.linspace(0, duration, num=n_frames)

    plt.figure(figsize=(15, 5))
    plt.plot(times, signal_array)
    plt.title("Audio signal")
    plt.ylabel("Signal Wave")
    plt.xlabel("Time (s)")
    plt.xlim(0, duration)
    plt.show()
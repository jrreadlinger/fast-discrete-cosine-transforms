import numpy as np
import os

def save_1d_signals(output_dir="data"):
    n = 64  # signal length

    # Step function: 0s followed by 1s
    step = np.concatenate([np.zeros(n // 2), np.ones(n // 2)])

    # Sine wave: 1 full period
    sine = np.sin(2 * np.pi * np.arange(n) / n)

    # Impulse: 1 at the center, 0 elsewhere
    impulse = np.zeros(n)
    impulse[n // 2] = 1

    # Save to .npy files
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "step_signal.npy"), step)
    np.save(os.path.join(output_dir, "sine_signal.npy"), sine)
    np.save(os.path.join(output_dir, "impulse_signal.npy"), impulse)

if __name__ == "__main__":
    save_1d_signals()
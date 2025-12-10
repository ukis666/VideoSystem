# Log-mel spectrogram patch

builtin_interfaces/Time stamp
uint32 sample_rate        # input sample rate, e.g. 16000
uint16 n_mels             # e.g. 64
uint16 n_frames           # time frames in this patch
float32 hop_length_s      # frame hop in seconds, e.g. 0.01
float32 win_length_s      # window length in seconds, e.g. 0.025
float32[] data            # row-major [n_frames * n_mels]

# Awesome Text-To-Speech Generation
This repository is prepared to present representative text-to-speech (TTS) systems. 

# Depployability
Results are calculated by generating the provided samples on consumer-level hardware. All experiments are conducted on an NVIDIA RTX 4060 laptop GPU. The reported results include the **full inference pipeline**, covering text tokenization, spectrogram or token generation, and waveform generation. For models that require a separate vocoder, the vocoder inference time is also included. Reported GPU memory usage is the **peak memory consumption** during the entire inference process.


| Model | Year | CUDA Memory | RTF | Deployability |
|-------|------|-------------|-----| ------------- | 
| Tacotron 1 & 2 | 2017-DEC | 1722.72 (MB) | 7.79 | 2.21|
| FastSpeech 1 & 2 | 2020-JUN | 443.29 (MB) | 8.01 | 4.21 |
| Glow-TTS | 2020-MAY | 8077.96 (MB) | 2.87  | -1.46 |
| VITS | 2021-JUN | 343.41 (MB) | 20.16 | 5.91 |
| E2-TTS | 2024-JUN | 3200 (MB) | 0.14 | -4.50 |
| F5-TTS | 2025-JUL | 2200 (MB) | 0.16 | -3.72 |
| CosyVoice | 2024-JUL | 2097.67 (MB) | 0.27 | -2.94 |
| IndexTTS | 2025-FEB | 1567.35 (MB) | 2.79 | 0.86 | 

# Speech Samples
TBA
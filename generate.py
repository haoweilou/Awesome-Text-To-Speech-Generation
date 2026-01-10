import time
import torch
from scipy.io.wavfile import write
from tts_loader import tacotron2,fastspeech2,glowtts
import torchaudio
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

with open("./text.txt", "r", encoding="utf-8") as f:
    texts = f.readlines()

texts = [text.strip() for text in texts if text.strip()]
model_name = "E2-TTS"
if model_name == "tacotron 2":
    tts,vocoder,utils = tacotron2()
elif model_name == "FastSpeech 2":
    tts,vocoder = fastspeech2()
elif model_name == "F5-TTS":
    import subprocess
elif model_name == "E2-TTS":
    import subprocess
    
cuda_max_mem = 0
total_inference_time = 0
frame_length = 0
for i,text in enumerate(texts):
    torch.cuda.reset_peak_memory_stats(device=None)
    print(i,text)
    start_time = time.time()
    if model_name == "tacotron 2":
        sequences, lengths = utils.prepare_input_sequence([text])
        with torch.no_grad():
            mel, _, _ = tts.infer(sequences, lengths)
            audio = vocoder.infer(mel)
        frame_length += mel.shape[2] # number of generated frames
    elif model_name == "FastSpeech 2":
        with torch.no_grad():
            mel, durations, pitch, energy = tts.encode_text(
                [text],
                pace=1.0,        # scale up/down the speed
                pitch_rate=1.0,  # scale up/down the pitch
                energy_rate=1.0, # scale up/down the energy
            )
            waveforms = vocoder.decode_batch(mel).detach().cpu()
        frame_length += mel.shape[2] # number of generated frames
    elif model_name == "F5-TTS":
        # minus the time of subprocess of loading (approximately 19.98 - 2.47 = 17.51s) loading time
        cmd = [
            "f5-tts_infer-cli",
            "--model", "F5TTS_Base",
            "--ref_audio", r"E:/combine/000000.wav",
            "--ref_text",
            "printing, in the only sense with which we are at present concerned,  differs from most if not from all the arts and crafts represented in the exhibition",
            "--gen_text",
            text,
            "-w", 
            f"C:/Users/Haowei/Desktop/Awesome-Text-To-Speech-Generation/samples/{model_name}/{i:02d}.wav"
        ]
        subprocess.run(cmd, check=True)
    elif model_name == "E2-TTS":
        # minus the time of subprocess of loading (approximately 20.60 - 3.26 = 17.34s) loading time
        cmd = [
            "f5-tts_infer-cli",
            "--model", "E2TTS_Base",
            "--ref_audio", r"E:/combine/000000.wav",
            "--ref_text",
            "printing, in the only sense with which we are at present concerned,  differs from most if not from all the arts and crafts represented in the exhibition",
            "--gen_text",
            text,
            "-w", 
            f"C:/Users/Haowei/Desktop/Awesome-Text-To-Speech-Generation/samples/{model_name}/{i:02d}.wav"
        ]
        subprocess.run(cmd, check=True)
        
    end_time = time.time()
    cuda_max_mem = max(cuda_max_mem, torch.cuda.max_memory_allocated(device=None))
    # saving time is not included in inference time
    if model_name == "tacotron 2":
        audio_numpy = audio[0].data.cpu().numpy()
        rate = 22050
        write(f"./samples/{model_name}/{i:02d}.wav", rate, audio_numpy)
    elif model_name == "FastSpeech 2":
        torchaudio.save(f"./samples/{model_name}/{i:02d}.wav", waveforms.squeeze(1), 22050)
   
    inference_time = end_time - start_time
    total_inference_time += inference_time
    
print("Model:", model_name)
print("Max CUDA memory allocated:", cuda_max_mem / 1024**2, "MB")
print("Total inference time:", total_inference_time)
print("Total generated frame length:", frame_length) # this is used to calculate the hz of feature
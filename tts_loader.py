import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def tacotron2():
    tacotron2 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tacotron2', model_math='fp16')
    tacotron2 = tacotron2.to(device)
    tacotron2.eval()
    
    waveglow = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_waveglow', model_math='fp16')
    waveglow = waveglow.remove_weightnorm(waveglow)
    waveglow = waveglow.to(device)
    waveglow.eval()
    utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tts_utils')
    return tacotron2, waveglow, utils

def fastspeech2():
    from speechbrain.inference.TTS import FastSpeech2
    from speechbrain.inference.vocoders import HIFIGAN
    fastspeech2 = FastSpeech2.from_hparams(source="speechbrain/tts-fastspeech2-ljspeech", savedir="pretrained_models/tts-fastspeech2-ljspeech", run_opts={"device": "cuda"})
    hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir="pretrained_models/tts-hifigan-ljspeech", run_opts={"device": "cuda"})
    # fastspeech2 = fastspeech2.to(device)
    # hifi_gan = hifi_gan.to(device)
    # fastspeech2.g2p.to("cpu")
    fastspeech2.eval()
    hifi_gan.eval()
    return fastspeech2, hifi_gan

def glowtts():
    # waveglow_path = './ckp/waveglow_256channels_ljs_v3.pt'
    # waveglow = torch.load(waveglow_path, map_location="cpu", weights_only=False)["model"]
    # waveglow = waveglow.remove_weightnorm(waveglow)
    # _ = waveglow.cuda().eval()
    waveglow = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_waveglow', model_math='fp16')
    waveglow = waveglow.remove_weightnorm(waveglow)
    waveglow = waveglow.to(device)
    waveglow.eval()
    from glow_tts import utils, models
    from glow_tts.text.symbols import symbols
    from glow_tts.text import text_to_sequence, cmudict
    # If you are using your own trained model
    model_dir = "./ckp/pretrained.pth"
    hps = utils.get_hparams_from_dir(model_dir)
    checkpoint_path = utils.latest_checkpoint_path(model_dir)

    model = models.FlowGenerator(
        len(symbols) + getattr(hps.data, "add_blank", False),
        out_channels=hps.data.n_mel_channels,
        **hps.model).to("cuda")

    utils.load_checkpoint(checkpoint_path, model)
    model.decoder.store_inverse() # do not calcuate jacobians for fast decoding
    _ = model.eval()

    cmu_dict = cmudict.CMUDict(hps.data.cmudict_path)
    return model, waveglow,hps, cmu_dict,text_to_sequence
# Run TTS with text input

from scipy.io.wavfile import write

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

path_to_xtts_model = "/Users/might/Library/Application Support/tts/tts_models--multilingual--multi-dataset--xtts_v2"

config = XttsConfig()
config.load_json(f"{path_to_xtts_model}/config.json")
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir=path_to_xtts_model, eval=True)
# model.cuda()

full_audio, full_tokens = model.preprocess(
    text="It took me quite a long time to develop a voice and now that I have it I am not going to be silent.",
    config=config,
    speaker_wav="/Users/might/Projects/datamonsters/speechka/xtts_server/xtts-server/speakers/male_1.wav",
    gpt_cond_len=3,
    language="en",
)

wavs = model.forward_inference(full_audio, full_tokens)
waws = wavs.cpu().numpy()

for i in range(waws.shape[0]):
    write(filename=f'serving/test_audio/test_{i}.wav', rate=24000, data=waws[i])

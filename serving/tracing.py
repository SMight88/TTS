import logging
import os
from typing import Literal

import torch

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

logging.basicConfig(level=logging.INFO)


class SpeakerEncoderWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, audio):
        return self.model.forward(
            audio,
            l2_norm=True
        )


class GPTGetStyleEmbWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, mel):
        return self.model.get_style_emb(
            mel,
            None
        )


class GPTGenerateWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, gpt_cond_latent, text_tokens):
        params = {
            'do_sample': True,
            'input_tokens': None,
            'length_penalty': 1.0,
            'num_beams': 1,
            'num_return_sequences': 1,
            'output_attentions': False,
            'repetition_penalty': 5.0,
            'temperature': 0.75,
            'top_k': 50,
            'top_p': 0.85
        }
        # gpt_cond_latent = inputs[0]
        # text_tokens = inputs[1]
        return self.model.generate(
            gpt_cond_latent,
            text_tokens,
            **params
        )


class Tracing:
    def __init__(
            self,
            path_to_model,
            path_to_traced_models,
            export_format: Literal['torchscript', 'onnx'] = 'onnx'
    ):
        self.path_to_model = path_to_model
        self.path_to_traced_models = path_to_traced_models
        self.export_format = export_format
        self.config = self.get_config()
        self.model = self.get_model()
        self.load_checkpoint()
        self.trace_audio_input = torch.load('serving/tensors/audio_16k.pt')
        self.trace_mel_input = torch.load('serving/tensors/mel_chunk.pt')
        self.trace_cond_latents_input = torch.load('serving/tensors/gpt_cond_latent.pt')
        self.trace_text_input = torch.load('serving/tensors/text_tokens.pt')

        # self.trace_audio_len = 300000
        # self.trace_mel_len = 500
        # self.trace_text_len = 100
        # self.trace_audio_input = torch.randn(
        #     (1, self.trace_audio_len), dtype=torch.float32, requires_grad=False, device=self.model.device
        # )
        # self.trace_mel_input = torch.rand(
        #     (1, 80, self.trace_mel_len), dtype=torch.float32, requires_grad=False, device=self.model.device
        # )
        # self.trace_cond_latents_input = torch.randn(
        #     (1, 32, 1024), dtype=torch.float32, requires_grad=False, device=self.model.device
        # )
        # self.trace_text_input = torch.randint(
        #     0, 200, (1, self.trace_text_len), dtype=torch.int32, requires_grad=False, device=self.model.device
        # )

    def get_config(self):
        config = XttsConfig()
        config_path = os.path.join(self.path_to_model, 'config.json')
        config.load_json(config_path)
        return config

    def get_model(self):
        return Xtts.init_from_config(self.config)

    def load_checkpoint(self):
        logging.info('Checkpoints loading...')
        self.model.load_checkpoint(self.config, checkpoint_dir=self.path_to_model, eval=True)

    @staticmethod
    def onnx_save(onnx_program, model_path):
        onnx_program.save(model_path)

    @staticmethod
    def jit_save(traced_model, model_path):
        torch.jit.save(traced_model, model_path)

    def export_and_save(self, model, example_inputs, model_name):
        model_path = os.path.join(self.path_to_traced_models, model_name)
        if self.export_format == 'torchscript':
            model_path += '.pt'
            traced_model = torch.jit.trace(
                model,
                example_inputs
            )
            self.jit_save(traced_model, model_path)
        elif self.export_format == 'onnx':
            model_path += '.onnx'
            if isinstance(example_inputs, tuple):
                pass
            onnx_program = torch.onnx.dynamo_export(
                model,
                example_inputs
            )
            self.onnx_save(onnx_program, model_path)
            # torch.onnx.export(
            #     model,  # model being run
            #     example_inputs,  # model input (or a tuple for multiple inputs)
            #     model_path,  # where to save the model (can be a file or file-like object)
            #     export_params=True,  # store the trained parameter weights inside the model file
            #     opset_version=18,  # the ONNX version to export the model to
            #     do_constant_folding=True,  # whether to execute constant folding for optimization
            #     input_names=['input'],  # the model's input names
            #     output_names=['output'],  # the model's output names
            #     dynamic_axes={
            #         'input': {1: 'audio_len'},  # variable length axes
            #         'output': {1: 'audio_len'}
            #     }
            # )

    def trace_speaker_encoder(self):
        logging.info('Tracing speaker_encoder...')
        wrapped_model = SpeakerEncoderWrapper(self.model.hifigan_decoder.speaker_encoder)
        model_name = 'speaker_encoder'
        self.export_and_save(wrapped_model, self.trace_audio_input, model_name)

    def trace_gpt_get_style_emb(self):
        logging.info('Tracing gpt.get_style_emb...')
        wrapped_model = GPTGetStyleEmbWrapper(self.model.gpt)
        model_name = 'gpt_get_style_emb'
        self.export_and_save(wrapped_model, self.trace_mel_input, model_name)

    def trace_gpt_generate(self):
        logging.info('Tracing gpt.generate...')
        wrapped_model = GPTGenerateWrapper(self.model.gpt)
        model_name = 'gpt_generate'
        self.export_and_save(
            wrapped_model,
            example_inputs=(self.trace_cond_latents_input, self.trace_text_input),
            model_name=model_name
        )

    def trace(self):
        logging.info(f'Tracing into {self.export_format} format...')
        # self.trace_speaker_encoder()
        # self.trace_gpt_get_style_emb()
        self.trace_gpt_generate()


if __name__ == '__main__':
    tracer = Tracing(
        path_to_model='/Users/might/Library/Application Support/tts/tts_models--multilingual--multi-dataset--xtts_v2',
        path_to_traced_models='/Users/might/Downloads/xtts_traced',
        export_format='onnx'
    )
    tracer.trace()

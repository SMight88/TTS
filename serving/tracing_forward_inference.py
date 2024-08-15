import argparse
import logging
import os
from typing import Literal

import pytorch_lightning as pl
import torch

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

logging.basicConfig(level=logging.INFO)


class XttsWrapper(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, audio, text_tokens):
        return self.model.forward_inference(
            audio,
            text_tokens
        )


class Tracing:
    def __init__(
            self,
            path_to_model: str,
            path_to_traced_models: str,
            export_format: Literal['torchscript', 'onnx'] = 'onnx',
            use_tensors: bool = False
    ):
        self.path_to_model = path_to_model
        self.path_to_traced_models = path_to_traced_models
        self.export_format = export_format
        self.config = self.get_config()
        self.model = self.get_model()
        self.load_checkpoint()
        if use_tensors:
            logging.info(f'Use predefined tensors for tracing instead of random tensors.')
            self.trace_audio_input = torch.load('serving/audio_16k.pt').to(self.model.device)
            self.trace_text_input = torch.load('serving/text_tokens.pt').to(self.model.device)
        else:
            self.trace_audio_len = 300000
            self.trace_text_len = 100
            self.trace_audio_input = torch.randn(
                (1, self.trace_audio_len), dtype=torch.float32, requires_grad=False, device=self.model.device
            )
            self.trace_text_input = torch.randint(
                0, 200, (1, self.trace_text_len), dtype=torch.int32, requires_grad=False, device=self.model.device
            )

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
        if torch.cuda.is_available():
            self.model.cuda()

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
            if isinstance(model, pl.LightningModule):
                model.to_torchscript(
                    file_path=model_path,
                    method="trace",
                    example_inputs=tuple(example_inputs.values())
                )
            else:
                traced_model = torch.jit.trace(
                    model,
                    example_kwarg_inputs=example_inputs
                )
                self.jit_save(traced_model, model_path)
        elif self.export_format == 'onnx':
            model_path += '.onnx'
            # onnx_program = torch.onnx.dynamo_export(
            #     model,
            #     **example_inputs
            # )
            # self.onnx_save(onnx_program, model_path)
            torch.onnx.export(
                model=model,  # model being run
                args=tuple(example_inputs.values()),  # model input (or a tuple for multiple inputs)
                f=model_path,  # where to save the model (can be a file or file-like object)
                export_params=True,  # store the trained parameter weights inside the model file
                verbose=True,
                opset_version=15,  # the ONNX version to export the model to
                do_constant_folding=True,  # whether to execute constant folding for optimization
                input_names=['input_0', 'input_1'],  # the model's input names
                output_names=['output'],  # the model's output names
                dynamic_axes={
                    'input_0': {0: 'batch_size', 1: 'audio_len'},
                    'input_1': {0: 'batch_size', 1: 'text_len'},
                    'output': {0: 'batch_size', 1: 'audio_len'}
                }
            )

    def trace(self):
        logging.info(f'Tracing into {self.export_format} format...')
        model_name = "xtts"
        wrapped_model = XttsWrapper(self.model)
        self.export_and_save(
            model=wrapped_model,
            example_inputs={
                'audio': self.trace_audio_input,
                'text_tokens': self.trace_text_input
            },
            model_name=model_name
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Tracing XTTS model.'
    )

    parser.add_argument(
        '--path_to_model', '-p',
        dest='path_to_model',
        nargs='?',
        type=str,
        help='Path to folder with XTTS checkpoints.',
        default='/Users/might/Library/Application Support/tts/tts_models--multilingual--multi-dataset--xtts_v2'
    )

    parser.add_argument(
        '--path_to_traced_models', '-s',
        dest='path_to_traced_models',
        nargs='?',
        type=str,
        help='Path to folder where traced model should be exported.',
        default='/Users/might/Downloads/xtts_traced'
    )

    parser.add_argument(
        '--export_format', '-e',
        dest='export_format',
        nargs='?',
        type=str,
        choices=['torchscript', 'onnx'],
        help='Export format.',
        default='onnx'
    )

    parser.add_argument(
        '--use_tensors',
        dest='use_tensors',
        help='If specified, then use predefined tensors instead of random input tensors.',
        action='store_true'
    )

    args = parser.parse_args()

    tracer = Tracing(
        path_to_model=args.path_to_model,
        path_to_traced_models=args.path_to_traced_models,
        export_format=args.export_format,
        use_tensors=args.use_tensors
    )
    tracer.trace()

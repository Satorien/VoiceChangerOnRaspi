from model import Net
import torch
import torchaudio
import pyaudio
import numpy as np
import argparse
import json
from utils import glob_audio_files
from tqdm import tqdm
import threading
import queue

def load_model(checkpoint_path, config_path):
    with open(config_path) as f:
        config = json.load(f) #dict型として読み込み
    model = Net(**config['model_params']) 
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu')['model'])
    return model, config['data']['sr']

# def read_thread(stream, chunk_size, data_queue):
#     while True:
#         data = np.frombuffer(stream.read(chunk_size), dtype=np.float32)
#         data_queue.put(data)

# def infer_thread(data_queue, model, stream_out):
#     with torch.inference_mode():
#         enc_buf, dec_buf, out_buf = model.init_buffers(1, torch.device('cpu'))
#         if hasattr(model, 'convnet_pre'):
#             convnet_pre_ctx = model.convnet_pre.init_ctx_buf(1, torch.device('cpu'))
#         else:
#             convnet_pre_ctx = None
#         while True:
#             data = data_queue.get()
#             tensor_data = torch.tensor(data)
#             if torch.max(torch.abs(tensor_data)) > 0.1:
#                 transformed_chunk, enc_buf, dec_buf, out_buf, convnet_pre_ctx = model(
#                     tensor_data.unsqueeze(0).unsqueeze(0), enc_buf, dec_buf, out_buf,
#                     convnet_pre_ctx, pad=(not model.lookahead))
#                 stream_out.write(transformed_chunk.squeeze(0).numpy().tobytes())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', '-p', type=str, default='llvc_models/models/checkpoints/llvc/G_500000.pth', help='Path to LLVC checkpoint file')
    parser.add_argument('--config_path', '-c', type=str, default='experiments/llvc/config.json', help='Path to LLVC config file')
    parser.add_argument('--chunk_factor', '-n', type=int, default=2048, help='Chunk factor')
    args = parser.parse_args()
    model, sr = load_model(args.checkpoint_path, args.config_path)
    # 量子化用の設定
    quantization_config = torch.ao.quantization.get_default_qat_qconfig('fbgemm')
    model.qconfig = quantization_config
    torch.backends.quantized.engine = 'fbgemm'
    quantized_model = torch.quantization.quantize_dynamic(model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8)
    
    # # モデルのトラバースとqconfigの変更
    # for module_name, module in model.named_modules():
    #     if isinstance(module, torch.nn.ConvTranspose2d):
    #         # ConvTransposeモジュールのqconfigをNoneに設定
    #         module.qconfig = None

    input_device_index = 1          ###要チェック
    output_device_index = 3
    sample_rate = 44100

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt8, channels=1, rate=sample_rate, input=True, frames_per_buffer=args.chunk_factor, output=True)

    with torch.inference_mode():
        enc_buf, dec_buf, out_buf = model.init_buffers(1, torch.device('cpu'))
        if hasattr(model, 'convnet_pre'):
            convnet_pre_ctx = model.convnet_pre.init_ctx_buf(1, torch.device('cpu'))
        else:
            convnet_pre_ctx = None

        while True:
            data = np.frombuffer(stream.read(args.chunk_factor), dtype=np.int8)
            data = data / max(np.max(np.abs(data)),1)
            data = torch.Tensor(data)
            data = torchaudio.transforms.Resample(sample_rate, sr, dtype=torch.int8)(data)
            quantized_data = torch.quantize_per_tensor(data, scale=1.0, zero_point=0, dtype=torch.quint8)
            transformed_chunk, enc_buf, dec_buf, out_buf, convnet_pre_ctx = quantized_model(
                quantized_data.unsqueeze(0).unsqueeze(0), enc_buf, dec_buf, out_buf,
                    convnet_pre_ctx, pad=(not model.lookahead))
            stream.write(transformed_chunk.squeeze(0).numpy().tobytes())


if __name__ == '__main__':
    main()

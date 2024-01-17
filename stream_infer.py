from model import Net
import torch
import torchaudio
import pyaudio
import numpy as np
import argparse
import json
from utils import glob_audio_files
from tqdm import tqdm

def load_model(checkpoint_path, config_path):
    with open(config_path) as f:
        config = json.load(f) #dict型として読み込み
    model = Net(**config['model_params']) 
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu')['model'])
    return model, config['data']['sr']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', '-p', type=str, default='llvc_models/models/checkpoints/llvc/G_500000.pth', help='Path to LLVC checkpoint file')
    parser.add_argument('--config_path', '-c', type=str, default='experiments/llvc/config.json', help='Path to LLVC config file')
    parser.add_argument('--chunk_factor', '-n', type=int, default=2048, help='Chunk factor')
    args = parser.parse_args()
    model, sr = load_model(args.checkpoint_path, args.config_path)
    
    input_device_index = 1          ###要チェック
    output_device_index = 3

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32, channels=1, rate=sr, input=True, frames_per_buffer=args.chunk_factor, output=True)

    with torch.inference_mode():
        enc_buf, dec_buf, out_buf = model.init_buffers(1, torch.device('cpu'))
        if hasattr(model, 'convnet_pre'):
            convnet_pre_ctx = model.convnet_pre.init_ctx_buf(1, torch.device('cpu'))
        else:
            convnet_pre_ctx = None

        while True:
            data = stream.read(args.chunk_factor)
            array_data = np.frombuffer(data, dtype=np.float32)
            tensor_data = torch.tensor(array_data, dtype=torch.float32)
            transformed_chunk, enc_buf, dec_buf, out_buf, convnet_pre_ctx = model(
                tensor_data.unsqueeze(0).unsqueeze(0), enc_buf, dec_buf, out_buf,
                    convnet_pre_ctx, pad=(not model.lookahead))
            stream.write(transformed_chunk.squeeze(0).numpy().tobytes())


if __name__ == '__main__':
    main()

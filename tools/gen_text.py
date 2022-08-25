# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import os
import soundfile
from tqdm import tqdm

import paddle
from paddlespeech.cli import ASRExecutor

def check_duration(audio_file: str):
    audio, audio_sample_rate = soundfile.read(
        audio_file, dtype="int16", always_2d=True)
    audio_duration = audio.shape[0] / audio_sample_rate
    max_duration = 50.0
    if audio_duration >= max_duration:
        return False
    return True

def process(path, lang, sr):
    if os.path.exists(path):
        files=os.listdir(path)
    else:
        print('this path not exist')
    asr_executor = ASRExecutor()
    for file in tqdm(files):
        if file.endswith('.wav'):
            print(file)
            file_name = os.path.splitext(file)[0]
            if os.path.exists(os.path.join(path, file_name + ".txt")):
                print("skip ", file_name)
                continue
            try:
                if not check_duration(os.path.join(path, file)):
                    raise Exception("Audio file more then 50 seconds.")
                text = asr_executor(
                    model='conformer_wenetspeech',
                    lang=lang,
                    sample_rate=sr,
                    config=None,  # Set `config` and `ckpt_path` to None to use pretrained model.
                    ckpt_path=None,
                    audio_file=os.path.join(path, file),
                    device=paddle.get_device())
                print('{} Result: \n{}'.format(file, text))

                with open(os.path.join(path, file_name + ".txt"), "w") as f:
                    f.write(text)
                f.close()
            except ValueError as e:
                print("cannot recognize")
                os.replace(os.path.join(path, file), os.path.join(path, "../unrecognized", file))
            except Exception as e:
                print(f"Error occur: {e}")
                os.replace(os.path.join(path, file), os.path.join(path, "../unrecognized", file))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--lang", type=str, default='zh')
    parser.add_argument("--sr", type=int, default=16000)
    args = parser.parse_args()

    # Check unrecognized exists or not
    isExist = os.path.exists(os.path.join(args.path, "../unrecognized"))
    if not isExist:
        os.makedirs(os.path.join(args.path, "../unrecognized"))
        print("unrecognized directory is created!")
    process(args.path, args.lang, args.sr)
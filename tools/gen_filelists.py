import os, tqdm, random

txt_str = []
# punc_txt_str = []


# 路径参数
speaker = "haruka"
p_wav_temp = r"C:\Users\Lisp\Downloads\other_file\VTuberTalk\data\wav_temp"
p_split = rf"{p_wav_temp}\{speaker}\split"


for file in tqdm.tqdm(os.listdir(p_split)):
    if file.endswith(".txt"): # and not file.endswith("_punc.txt"):
        name, suffix = file.split(".")

        with open(rf"{p_split}\{name}.txt", "r", encoding="utf8") as f:
            txt = f.read()
        # with open(rf"{p_split}\{name}_punc.txt", "r", encoding="utf8") as f:
        #     punc_txt = f.read()
        
        # 文本内容
        txt_str.append(f"biaobei/{name}.wav|{txt}")
        # punc_txt_str.append(f"biaobei/{name}.wav|{punc_txt}")


with open(rf"{p_wav_temp}\{speaker}\filelist.txt", "w", encoding="utf8") as f:
    f.write("\n".join(txt_str))
# with open(rf"{wav_temp}\{speaker}\punc_filelist.txt", "w", encoding="utf8") as f:
#     f.write("\n".join(punc_txt_str))



with open(rf"{p_wav_temp}\{speaker}\filelist.txt", "r", encoding="utf8") as f:
    files = f.read().splitlines()
random.shuffle(files)

with open(rf"{p_wav_temp}\{speaker}\train_filelist.txt", "w", encoding="utf8") as f:
    f.write("\n".join(files[:int(len(files)*0.95)]))
with open(rf"{p_wav_temp}\{speaker}\val_filelist.txt", "w", encoding="utf8") as f:
    f.write("\n".join(files[int(len(files)*0.95):]))


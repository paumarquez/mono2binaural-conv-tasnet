import os
import random
import json

PATH = './hdf5/binaural_audios'
random.seed(1)
audios = os.listdir(PATH)
train_prop = 0.7
val_prop = 0.2
test_prop = 1 - train_prop - val_prop
train_idx = int(len(audios)*train_prop)
val_idx = train_idx + int(len(audios)*val_prop)
random.shuffle(audios)
train = audios[:train_idx]
val = audios[train_idx:val_idx]
test = audios[val_idx:]

with open('./hdf5/splits/split2.json', 'w') as fd:
    json.dump({
        'train': train,
        'val': val,
        'test': test,
    }, fd)

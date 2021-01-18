import h5py
import json
import os
path = 'hdf5/their_splits/splits/split1'
filenames= ['train.h5', 'val.h5', 'test.h5']
key_names = ['train', 'val','test']
obj = {}
for i, filename in enumerate(filenames):
    with h5py.File(os.path.join(path, filename), "r") as f:
        a_group_key = list(f.keys())[0]
        data = list(f[a_group_key])
        data = list(map(str,data))
        new_data = []
        for f in data:
            new_data.append([sub_filename[:-1] for sub_filename in f.split('/') if '.wav' in sub_filename][0])
        obj[key_names[i]] = new_data
with open('./hdf5/splits/split1.json', 'w') as fd:
    json.dump(obj, fd)

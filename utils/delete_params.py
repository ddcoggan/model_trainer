# /usr/bin/python
# Created by David Coggan on 2022 10 21

# deletes model params except first, last and every n epochs
import os, glob
saveEvery = 4
for folder, subfolders, subfiles in os.walk('DNN/data'):
    for file in sorted(subfiles):
        if file.endswith('.pt'):
            epoch = int(file[:3])
            if epoch > 0 and epoch % saveEvery != 0 and f"{folder}/{file}" != sorted(glob.glob(f"{folder}/*.pt"))[-1]:
                print(f"removing {folder}/{file}")
                os.remove(f"{folder}/{file}")
import csv
import faiss
from sklearn.preprocessing import normalize
import json
import random
import math
import time
import struct
import glob
import numpy as np
import os
import tensorflow as tf
from keras.applications.mobilenet_v2 import preprocess_input
import keras.layers as layers
from keras.models import Model


def preprocess(img_path, input_shape):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=input_shape[2])
    img = tf.image.resize(img, input_shape[:2])
    img = preprocess_input(img)
    return img


def main():
    batch_size = 100
    input_shape = (224, 224, 3)
    base = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                             include_top=False,
                                             weights='imagenet')
    base.trainable = False
    model = Model(inputs=base.input,
                  outputs=layers.GlobalAveragePooling2D()(base.output))

    root_dir = "이미지 존재하는 루트 디렉토리"
    images = []
    for(dirpath, dirnames, filenames) in os.walk(root_dir):
        for filename in filenames:
            if ".png" in filename or ".jpg" in filename:
                images.append(dirpath + "/" + filename)
    fnames = images

    list_ds = tf.data.Dataset.from_tensor_slices(fnames)
    ds = list_ds.map(lambda x: preprocess(
        x, input_shape), num_parallel_calls=-1)
    dataset = ds.batch(batch_size).prefetch(-1)

    with open('fvecs.bin', 'wb') as f:
        for batch in dataset:
            fvecs = model.predict(batch)

            fmt = f'{np.prod(fvecs.shape)}f'
            f.write(struct.pack(fmt, *(fvecs.flatten())))

    with open('fnames.txt', 'w') as f:
        f.write('\n'.join(fnames))


if __name__ == '__main__':
    main()


def dist2sim(d):
    return 1 - d / 2


def get_index(index_type, dim):
    if index_type == 'hnsw':
        m = 48
        index = faiss.IndexHNSWFlat(dim, m)
        index.hnsw.efConstruction = 128
        return index
    elif index_type == 'l2':
        return faiss.IndexFlatL2(dim)
    raise


def populate(index, fvecs, batch_size=1000):
    nloop = math.ceil(fvecs.shape[0] / batch_size)
    for n in range(nloop):
        s = time.time()
        index.add(
            normalize(fvecs[n * batch_size: min((n + 1) * batch_size, fvecs.shape[0])]))
        print(n * batch_size, time.time() - s)

    return index


def find_file_name(idx):
    if idx == -1:
        return 'None'
    with open('fnames.txt', 'r') as f:
        names = f.readlines()
    return names[idx].strip('\n').strip('\t')


def main():
    dim = 1280
    fvec_file = 'fvecs.bin'
    index_type = 'hnsw'
    #index_type = 'l2'

    # f-string 방식 (python3 이상에서 지원)
    index_file = f'{fvec_file}.{index_type}.index'

    fvecs = np.memmap(fvec_file, dtype='float32', mode='r').view(
        'float32').reshape(-1, dim)

    if os.path.exists(index_file):
        index = faiss.read_index(index_file)
        if index_type == 'hnsw':
            index.hnsw.efSearch = 256
    else:
        index = get_index(index_type, dim)
        index = populate(index, fvecs)
        faiss.write_index(index, index_file)
    print(index.ntotal)

    random.seed(2020)

    # random하게 쿼리 인덱스를 생성한다
    # 0부터 데이터 갯수 사이의 인덱스
    q_idx = [random.randint(0, fvecs.shape[0]) for _ in range(100)]
    q_idx = np.arange(0, 1)
    k = 10
    s = time.time()

    total = 0
    csv_file = open('result.csv', 'w', encoding='utf-8-sig', newline='')
    wr = csv.writer(csv_file)
    for source in range(index.ntotal):
        q_idx = [source]
        dists, idxs = index.search(normalize(fvecs[q_idx]), k)
        sim_files = []
        sim_scores = []
        for i, idx in enumerate(idxs[0]):
            if dists[0][i] <= 0.3 and idx != source:
                sim_scores.append(dists[0][i])
                sim_files.append(find_file_name(idx))
                total = total + 1

        if len(sim_files) > 0:
            wr.writerow([find_file_name(source)])
            for i, file in enumerate(sim_files):
                wr.writerow([sim_scores[i], file])

    print("length = " + str(len(sim_files)))
    csv_file.close()
    print(total)


if __name__ == '__main__':
    main()

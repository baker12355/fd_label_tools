import numpy as np
import os
import cv2
import json
import tqdm
import argparse
import threading
from retina_fd import RetinaDetector


def label(paths, model):
    for path in tqdm.tqdm(paths):
        img = cv2.imread(path, 1)
        bboxes, lmks = model.infer(img)
        struc = {'img_path':"", 'bboxes':[], 'lmks':[]}
        struc['img_path'] = path
        if len(bboxes) > 0:
            for bbox, lmk in zip(bboxes,lmks):
                struc['bboxes'].append(bbox.tolist())
                struc['lmks'].append(lmk.tolist())
        else:
            # No face detected.
            pass
        
        # Add result to list
        init.append(struc)
    return None

def run_fd_multi_thread_gpu(paths, thread):
    """Run retina face detector on gpu with multi-thread

    You should let all images in the same size for a consistent inference time.
    
    Args:
        input_dir: a folder to be label
        workers: number of threads
    Returns:
        Save labels to ./results/dirname.json
    """
    
    threads = []
    models = [RetinaDetector(i) for i in range(thread)]

    for i in range(thread):
        threads.append(threading.Thread(target=label, args=(paths[i::thread], models[i])))
        threads[i].start()
    for i in range(thread):
        threads[i].join()

def parse_args():
    parser = argparse.ArgumentParser(description='FD with multi-thread and multi-GPU')
    parser.add_argument('--input_dir', type=str)
    parser.add_argument('-j', '--workers', default=1, type=int)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    args = parse_args()
    root = args.input_dir
    thread = args.workers
    
    dataset = root.split('/')[-1]
    save_dir = 'results/%s.json'%(dataset)
    if os.path.isfile(save_dir):
        print('Attemp to overwrite an exist file: %s'%(save_dir))
        raise FileExistsError

    paths = []
    for r, dirs, files in os.walk(root):
        for file in files:
            # You should check your image extension
            paths.append(os.path.join(r, file))
    
    print('-'*50)
    print('root:', root)
    print('number of images:', len(paths))
    print('-'*50)
    
    init = []
    run_fd_multi_thread_gpu(paths, thread)
    
    print('length of lables:', len(init))
    
    init = sorted(init, key=lambda x: x['img_path'])
    
    with open(save_dir, 'w') as outfile:
        json.dump({'imgs': init}, outfile)

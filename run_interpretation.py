import os
import torch
import random
import torch.hub
import glob
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from consts import height, width, segment_count, class_counts, repo, batch_size, \
    nouns, verbs, frames_path_pattern, heads, base_models, device, random_iters, augm_fn_list, \
    fine_tune_verbs, fine_tune_val_split, fine_tune_epochs, fine_tune_head, fine_tune_base, fine_tune_lr, \
    trained_models_dir
from NarrowModel import NarrowModel
from KitchenDataset import KitchenDataset
from augmentations import get_4_augms_list, get_1_augms_list
from evaluate import init_metrics,evaluate


if __name__=="__main__":
    model=torch.load(os.path.join(trained_models_dir,'model.pth'))
    video_path='data/frames_b/cheese_put_1'
    train_dataset = KitchenDataset([video_path], height, width, segment_count, model,
                                   verbs_list=fine_tune_verbs,
                                   is_random=False, augm_fn=None)

    snippets,target = train_dataset[0]

    outputs=model(snippets)

    dbg=1

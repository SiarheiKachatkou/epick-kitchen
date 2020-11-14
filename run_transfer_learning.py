import os
import torch
import random
import torch.hub
import glob
import time
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from utils_snippets import clip_into_snippets, noun_verb_from_path
from consts import height, width, segment_count, class_counts, repo, batch_size, \
    nouns, verbs, frames_path_pattern, heads, base_models, device, random_iters, augm_fn_list, \
    fine_tune_verbs, fine_tune_val_split, fine_tune_epochs
from utils_visualization import show_snippet
from utils_snippets import normalize_inputs_for_model
from NarrowModel import NarrowModel
from KitchenDataset import KitchenDataset
from augmentations import get_4_augms_list, get_1_augms_list

def get_filtered_paths(root_path, verbs_list):
    p=[]
    for v in verbs_list:
        p+=glob.glob(os.path.join(root_path,f'*{v}*'))
    return p

if __name__=="__main__":

    perfs = {}
    train_val_video_paths = get_filtered_paths('data/frames_b',fine_tune_verbs)
    random.shuffle(train_val_video_paths)
    split=int(fine_tune_val_split*len(train_val_video_paths))
    val_videos_path=train_val_video_paths[:fine_tune_val_split]
    train_videos_path=train_val_video_paths[fine_tune_val_split:]

    test_video_paths=glob.glob('data/frames_a',fine_tune_verbs)

    head=heads[1]
    base_model=base_models[1]

    try:
        model = torch.hub.load(repo, head, class_counts, segment_count, 'RGB',
                               base_model=base_model,
                               pretrained='epic-kitchens', force_reload=True)
        model.eval()
        model.to(device)
    except:
        print(f'enable load {head} with {base_model}')
        raise NotImplementedError()

    train_dataset=KitchenDataset(train_videos_path,height,width,segment_count,model,is_random=True,augm_fn=get_4_augms_list())
    train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)

    val_dataset=KitchenDataset(val_videos_path,height,width,segment_count,model,is_random=True,augm_fn=get_4_augms_list())
    val_loader=DataLoader(val_dataset,batch_size=batch_size,shuffle=False)

    test_dataset=KitchenDataset(test_video_paths,height,width,segment_count,model,is_random=False,augm_fn=None)
    test_loader=DataLoader(test_dataset,batch_size=batch_size,shuffle=False)


    metrics=pd.DataFrame()

    for epoch in range(fine_tune_epochs):
        for snippets,targets in train_loader:


            for random_iter in tqdm(range(random_iters)):
                for augm_fn in augm_fn_list:
                    for src_video_path in src_video_paths:
                        snippets = clip_into_snippets(src_video_path, height, width, segment_count, is_random=True,
                                                      augm_fn=augm_fn_list)

                        target_noun, target_verb = noun_verb_from_path(src_video_path)

                        inputs = normalize_inputs_for_model(snippets, model)

                        inputs = inputs.reshape((batch_size, -1, height, width))
                        inputs = inputs.to(device)

                        start = time.time()
                        with torch.no_grad():
                            verb_logits, noun_logits = model(inputs)
                        finish = time.time()
                        the_time = finish - start
                        update_performance(model_performance, noun_logits, verb_logits, nouns, verbs, target_noun,
                                           target_verb, the_time)

                        # show_snippet(snippets)

            finalize_performance(model_performance)
            print(model_performance)
            perfs.update({(head, base_model): model_performance})

    plot_performance(perfs, base_models, heads)

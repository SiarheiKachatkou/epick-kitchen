import os
import torch
import torch.hub
import glob
import time
import numpy as np
from tqdm import tqdm
from utils_snippets import clip_into_snippets, noun_verb_from_path
from consts import height, width, segment_count, class_counts, repo, batch_size, nouns, verbs
from utils_visualization import show_snippet
from performance_metric import update_performance, finalize_performance, init_model_performance_dict, \
    plot_performace, init_nan_model_performance_dict
from augmentations import get_4_augms_list, get_1_augms_list
from utils_snippets import normalize_inputs_for_model

base_models = ['resnet50', 'BNInception']
heads = ['TSN', 'TRN', 'MTRN', 'TSM']
device = 'cpu' #'cuda'

random_iters = 4
augm_fn_list = get_4_augms_list()

perfs = {}
src_video_paths = glob.glob('data/frames/*')

for head in heads:
    for base_model in base_models:

        try:
            model = torch.hub.load(repo, head, class_counts, segment_count, 'RGB',
                                   base_model=base_model,
                                   pretrained='epic-kitchens', force_reload=True)
            model.eval()
            model.to(device)

        except:
            print(f'enable load {head} with {base_model}')
            perfs.update({(head, base_model): init_nan_model_performance_dict()})
            continue

        model_performance = init_model_performance_dict()

        for random_iter in tqdm(range(random_iters)):
            for augm_fn in augm_fn_list:
                for src_video_path in src_video_paths:
                    snippets = clip_into_snippets(src_video_path, height, width, segment_count, is_random=True,
                                                  augm_fn=augm_fn)
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

plot_performace(perfs, base_models, heads)
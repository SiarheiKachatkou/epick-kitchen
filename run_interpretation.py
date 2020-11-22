import os
import cv2
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

def show_snippet(snippet,most_important_tuple,target_idx,predicted_idx, max_prob_degradation, channels=3):
    rows=2
    snippet=snippet.detach().cpu().numpy()
    images=[snippet[i*channels:(i+1)*channels] for i in range(len(snippet)//channels)]
    images=[cv2.normalize(np.transpose(im,(1,2,0)),None,0,255,cv2.NORM_MINMAX).astype(np.uint8) for im in images]
    for i,img in enumerate(images):
        plt.subplot(rows,len(images)//rows,i+1)
        plt.imshow(img)
        if i in most_important_tuple:
            plt.title('!')
        plt.axis('off')

    plt.suptitle(f'predicted={fine_tune_verbs[predicted_idx]} target={fine_tune_verbs[target_idx]} importance={max_prob_degradation}')


if __name__=="__main__":
    wide_model = torch.hub.load(repo, fine_tune_head, class_counts, segment_count, 'RGB',
                                base_model=fine_tune_base,
                                pretrained='epic-kitchens', force_reload=True)

    model=torch.load(os.path.join(trained_models_dir,'model.pth'))
    video_path='data/frames/hand_put'
    train_dataset = KitchenDataset([video_path], height, width, segment_count, wide_model,
                                   verbs_list=fine_tune_verbs,
                                   is_random=False, augm_fn=None)

    snippets,target = train_dataset[0]
    snippets = snippets.to(device)

    probs=torch.nn.functional.softmax(model(torch.unsqueeze(snippets,dim=0)))
    predicted=torch.argmax(probs)
    prob=probs[0, predicted].detach().cpu().numpy()

    channels=3
    max_prob_degradation=0
    most_influential_tuple=None
    for i in range(segment_count):
        for j in range(i+1,segment_count):
            snippets_permuted=snippets
            slice_list=[snippets[:i*channels],snippets[j*channels:(j+1)*channels],snippets[(i+1)*channels:j*channels],snippets[i*channels:(i+1)*channels],snippets[(j+1)*channels:]]
            slice_list=[s for s in slice_list if s.shape[0]!=0]
            snippets_permuted=torch.cat(slice_list,dim=0)
            probs = torch.nn.functional.softmax(model(torch.unsqueeze(snippets_permuted,dim=0)))
            permuted_prob=probs[0, predicted].detach().cpu().numpy()
            prob_degradation=prob-permuted_prob
            if prob_degradation>max_prob_degradation:
                max_prob_degradation=prob_degradation
                most_influential_tuple=(i,j)

    predicted=int(predicted.detach().cpu().numpy())
    show_snippet(snippets, most_influential_tuple, target, predicted, max_prob_degradation)
    plt.show()
    plt.savefig('plots/intepretation.png')

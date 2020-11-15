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
    fine_tune_verbs, fine_tune_val_split, fine_tune_epochs, fine_tune_head, fine_tune_base, fine_tune_lr
from NarrowModel import NarrowModel
from KitchenDataset import KitchenDataset
from augmentations import get_4_augms_list, get_1_augms_list
from evaluate import init_metrics,evaluate

def get_filtered_paths(root_path, verbs_list):
    p=[]
    for v in verbs_list:
        p+=glob.glob(os.path.join(root_path,f'*{v}*'))
    return p

if __name__=="__main__":

    fine_tune_class_idxs=[np.where(verbs.class_key==v)[0][0] for v in fine_tune_verbs]

    train_val_video_paths = get_filtered_paths('data/frames_b',fine_tune_verbs)
    random.shuffle(train_val_video_paths)
    split=int(fine_tune_val_split*len(train_val_video_paths))
    val_videos_path=train_val_video_paths[:split]
    train_videos_path=train_val_video_paths[split:]

    test_video_paths=get_filtered_paths('data/frames',fine_tune_verbs)

    try:
        wide_model = torch.hub.load(repo, fine_tune_head, class_counts, segment_count, 'RGB',
                               base_model=fine_tune_base,
                               pretrained='epic-kitchens', force_reload=True)

    except:
        print(f'enable load {fine_tune_head} with {fine_tune_base}')
        raise NotImplementedError()

    model = NarrowModel(wide_model=wide_model, verb_class_idxs=fine_tune_class_idxs)
    model.to(device)

    train_dataset=KitchenDataset(train_videos_path,height,width,segment_count,wide_model,verbs_list=fine_tune_verbs,
                                 is_random=True,augm_fn=get_4_augms_list())
    train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True,drop_last=False)

    val_dataset=KitchenDataset(val_videos_path,height,width,segment_count,wide_model,
                               verbs_list=fine_tune_verbs, is_random=True,augm_fn=get_4_augms_list())
    val_loader=DataLoader(val_dataset,batch_size=batch_size,shuffle=False,drop_last=False)

    test_dataset=KitchenDataset(test_video_paths,height,width,segment_count,wide_model,
                                verbs_list=fine_tune_verbs,is_random=False,augm_fn=None)
    test_loader=DataLoader(test_dataset,batch_size=batch_size,shuffle=False, drop_last=False)

    metrics=init_metrics(fine_tune_epochs)

    criterion=torch.nn.CrossEntropyLoss()

    metrics=evaluate(metrics,model,criterion, train_loader, val_loader, test_loader, epoch=0)

    verbs_linear_layer=list(model._wide_model.modules())[-1]
    optimizer = torch.optim.Adam(verbs_linear_layer.parameters(),lr=fine_tune_lr)
    for epoch in tqdm(range(fine_tune_epochs)):
        model.train()
        for snippets,targets in train_loader:
            optimizer.zero_grad()
            snippets=snippets.to(device)
            targets = targets.to(device)
            logits=model(snippets)
            loss=criterion(logits,targets)
            loss.backward()
            optimizer.step()

        metrics = evaluate(metrics, model, criterion, train_loader, val_loader, test_loader, epoch=epoch+1)

    print(metrics.head())
    metrics.to_csv('plots/fine_tune_metrics.csv')
    columns = metrics.columns
    for key in ['acc', 'loss']:
        cols = [c for c in columns if key in c]
        metrics[cols].plot()
        plt.title(key)
        plt.savefig(f'plots/fine_tune_{key}.png')

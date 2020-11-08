import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils_visualization import get_topK_words

def update_performance(models_performance, noun_logits, verb_logits, nouns, verbs, target_noun, target_verb, the_time):
    topk_verbs = get_topK_words(verb_logits, verbs['class_key'], k=5)
    topk_nouns = get_topK_words(noun_logits, nouns['class_key'], k=5)
    models_performance['verb_acc'].append(int(target_verb==topk_verbs[0]))
    models_performance['noun_acc'].append(int(target_noun == topk_nouns[0]))
    models_performance['verb_acc_top5'].append(int(target_verb in topk_verbs))
    models_performance['noun_acc_top5'].append(int(target_noun in topk_nouns))

    noun_label=torch.tensor([nouns[nouns['class_key'] == target_noun]['noun_id'].values[0]])
    noun_label=noun_label.to(noun_logits.device)
    noun_loss=torch.nn.functional.cross_entropy(noun_logits,noun_label)

    verb_label = torch.tensor([verbs[verbs['class_key'] == target_verb]['verb_id'].values[0]])
    verb_label = verb_label.to(verb_logits.device)
    verb_loss = torch.nn.functional.cross_entropy(verb_logits, verb_label)

    loss=(noun_loss+verb_loss)*0.5
    models_performance['loss'].append(loss.detach().cpu().numpy())

    models_performance['time'].append(the_time)


def finalize_performance(models_performance):
    for k,vals in models_performance.items():
        models_performance[k]=np.mean(vals)


model_performance_keys = ['noun_acc', 'verb_acc', 'noun_acc_top5', 'verb_acc_top5', 'time', 'loss']


def init_model_performance_dict():
    return {k: [] for k in model_performance_keys}


def init_nan_model_performance_dict():
    return {k: np.nan for k in model_performance_keys}


def plot_performace(perfs, base_models, heads):
    for metric_key in model_performance_keys:
        data = []
        for b in base_models:
            s = []
            for h in heads:
                s.append(perfs[(h, b)][metric_key])
            data.append(s)
        df = pd.DataFrame(data=data, columns=heads, index=base_models)
        df.plot.bar(rot=0)
        plt.savefig(os.path.join('plots', metric_key + '.png'))


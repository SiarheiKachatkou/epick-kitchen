import torch
import torch.hub
import glob
import time
import numpy as np
from utils_snippets import clip_into_snippets, noun_verb_from_path
from consts import height,width,segment_count,class_counts, repo, batch_size, nouns, verbs
from utils_visualization import get_topK_words, show_snippet

def update_performance(models_performance, noun_logits, verb_logits, nouns, verbs, target_noun, target_verb, the_time):
    topk_verbs = get_topK_words(verb_logits, verbs['class_key'], k=5)
    topk_nouns = get_topK_words(noun_logits, nouns['class_key'], k=5)
    models_performance['verb_acc'].append(int(target_verb==topk_verbs[0]))
    models_performance['noun_acc'].append(int(target_noun == topk_nouns[0]))
    models_performance['verb_acc_top5'].append(int(target_verb in topk_verbs))
    models_performance['noun_acc_top5'].append(int(target_noun in topk_nouns))

    noun_label=torch.tensor([nouns[nouns['class_key'] == target_noun]['noun_id'].values[0]])
    noun_loss=torch.nn.functional.cross_entropy(noun_logits,noun_label)

    verb_label = torch.tensor([verbs[verbs['class_key'] == target_verb]['verb_id'].values[0]])
    verb_loss = torch.nn.functional.cross_entropy(verb_logits, verb_label)

    loss=(noun_loss+verb_loss)*0.5
    models_performance['loss'].append(loss.detach().cpu().numpy()[0])

    models_performance['time'].append(the_time)


def finalize_performance(models_performance):
    for k,vals in models_performance.items():
        models_performance[k]=np.mean(vals)

base_model = 'resnet50'
tsn = torch.hub.load(repo, 'TSN', class_counts, segment_count, 'RGB',
                     base_model=base_model,
                     pretrained='epic-kitchens', force_reload=True)
trn = torch.hub.load(repo, 'TRN', class_counts, segment_count, 'RGB',
                     base_model=base_model,
                     pretrained='epic-kitchens')
mtrn = torch.hub.load(repo, 'MTRN', class_counts, segment_count, 'RGB',
                      base_model=base_model,
                      pretrained='epic-kitchens')
tsm = torch.hub.load(repo, 'TSM', class_counts, segment_count, 'RGB',
                     base_model=base_model,
                     pretrained='epic-kitchens')

# Show all entrypoints and their help strings
for entrypoint in torch.hub.list(repo):
    print(entrypoint)
    print(torch.hub.help(repo, entrypoint))

src_video_paths = glob.glob('data/frames/*')



for model in [tsn, trn, mtrn, tsm]:

    models_performance = {'noun_acc': [], 'verb_acc': [],'noun_acc_top5': [], 'verb_acc_top5': [], 'time': [], 'loss': []}

    for src_video_path in src_video_paths:

        print(f'video_path={src_video_path}')
        snippets = clip_into_snippets(src_video_path, height, width, segment_count)
        target_noun,target_verb=noun_verb_from_path(src_video_path)

        m = np.reshape(model.base_model.mean, (1, 1, 1, 3, 1, 1))
        s = np.reshape(model.base_model.std, (1, 1, 1, 3, 1, 1))

        inputs = torch.tensor((snippets/255-m)/s).float()
        inputs = inputs.reshape((batch_size, -1, height, width))

        start=time.time()
        verb_logits, noun_logits = model(inputs)
        finish = time.time()
        the_time=finish-start
        update_performance(models_performance, noun_logits, verb_logits, nouns, verbs, target_noun, target_verb, the_time)

        #show_snippet(snippets)
        dbg=1
    finalize_performance(models_performance)
    print(models_performance)


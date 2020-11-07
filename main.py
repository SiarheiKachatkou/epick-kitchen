import torch.hub
import pandas as pd
from split_video_into_frames import clip_into_input


src_video_path = 'data/Tiny KFC.mp4'

nouns = pd.read_csv('data/EPIC_noun_classes.csv')
verbs = pd.read_csv('data/EPIC_verb_classes.csv')

batch_size = 1
segment_count = 8
snippet_length = 1  # Number of frames composing the snippet, 1 for RGB, 5 for optical flow
snippet_channels = 3  # Number of channels in a frame, 3 for RGB, 2 for optical flow
height, width = 224, 224

snippets = clip_into_input(src_video_path,height,width, segment_count)

inputs = torch.tensor(snippets)

repo = 'epic-kitchens/action-models'

class_counts = (125, 352)

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


# The segment and snippet length and channel dimensions are collapsed into the channel
# dimension
# Input shape: N x TC x H x W
inputs = inputs.reshape((batch_size, -1, height, width))
for model in [tsn, trn, mtrn, tsm]:
    # You can get features out of the models
    features = model.features(inputs)
    # and then classify those features
    verb_logits, noun_logits = model.logits(features)

    # or just call the object to classify inputs in a single forward pass
    verb_logits, noun_logits = model(inputs)
    verb_idx=verb_logits.argmax(dim=1).detach().cpu().numpy()[0]
    noun_idx = noun_logits.argmax(dim=1).detach().cpu().numpy()[0]

    print(verb_logits.shape, noun_logits.shape)
    n=nouns.iloc[noun_idx]['nouns']
    print(f"noun = {n}")
    v=verbs.iloc[verb_idx]['verbs']
    print(f"verbs = {v}")


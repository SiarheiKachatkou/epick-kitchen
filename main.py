import torch.hub
import glob
from clip_into_snippets import clip_into_snippets
from consts import height,width,segment_count,class_counts, repo, batch_size, nouns, verbs
from utils_visualization import get_topK_words, show_snippet


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

src_video_paths = glob.glob('data/clips/*')
for model in [tsn, trn, mtrn, tsm]:
    for src_video_path in src_video_paths:

        print(f'video_path={src_video_path}')

        snippets = clip_into_snippets(src_video_path, height, width, segment_count)

        # The segment and snippet length and channel dimensions are collapsed into the channel
        # dimension
        # Input shape: N x TC x H x W
        inputs = torch.tensor(snippets)
        inputs = inputs.reshape((batch_size, -1, height, width))
        # You can get features out of the models
        features = model.features(inputs)
        # and then classify those features
        verb_logits, noun_logits = model.logits(features)

        # or just call the object to classify inputs in a single forward pass
        verb_logits, noun_logits = model(inputs)

        topk_verbs = get_topK_words(verb_logits,verbs['class_key'], k=5)
        topk_nouns = get_topK_words(noun_logits, nouns['class_key'], k=5)

        print(verb_logits.shape, noun_logits.shape)

        print(f"noun = {topk_nouns}")
        print(f"verbs = {topk_verbs}")
        show_snippet(snippets)


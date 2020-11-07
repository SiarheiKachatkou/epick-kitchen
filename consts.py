import pandas as pd

batch_size = 1
segment_count = 8
snippet_length = 1  # Number of frames composing the snippet, 1 for RGB, 5 for optical flow
snippet_channels = 3  # Number of channels in a frame, 3 for RGB, 2 for optical flow
height, width = 224, 224

repo = 'epic-kitchens/action-models'

class_counts = (125, 352)


nouns = pd.read_csv('data/EPIC_noun_classes.csv')
verbs = pd.read_csv('data/EPIC_verb_classes.csv')
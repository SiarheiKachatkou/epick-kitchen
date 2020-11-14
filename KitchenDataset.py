import torch
from utils_snippets import clip_into_snippets, normalize_inputs_for_model

class KitchenDataset(torch.utils.data.Dataset):
    def __init__(self, path_to_clips_list, height, width, segment_count, model_to_normalize_for, is_random=False, augm_fn=None):
        super().__init__()
        self._paths=path_to_clips_list
        self._height=height
        self._width=width
        self._segment_count=segment_count
        self._is_random=is_random
        self._augm_fn=augm_fn
        self._model_to_normalize_for=model_to_normalize_for

    def __getitem__(self, item):
        video_path=self._paths[item]
        snippets = clip_into_snippets(video_path, self._height, self._width, self._segment_count, self._is_random,
                                                      self._augm_fn)
        target_noun, target_verb = noun_verb_from_path(src_video_path)

        inputs = normalize_inputs_for_model(snippets, self._model_to_normalize_for)

        inputs = inputs.reshape((-1, self._height, self._width))
        return inputs,target_verb

    def __len__(self):
        return len(self._paths)

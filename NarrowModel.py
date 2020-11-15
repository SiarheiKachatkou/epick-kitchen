import torch

class NarrowModel(torch.nn.Module):
    def __init__(self, wide_model, verb_class_idxs):
        super().__init__()
        self._wide_model=wide_model
        self._verb_idxs=verb_class_idxs

    def forward(self, inputs):
        noun_logits, verb_logits = self._wide_model(inputs)
        return verb_logits[:, self._verb_idxs]

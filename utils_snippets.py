import os
import numpy as np
import cv2
import random
import torch


def _img_standardize(frame, frame_height, frame_width):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (frame_width, frame_height))
    return frame


def clip_to_frame_list(path_to_video, frame_height, frame_width):

    frame_list=[]
    if os.path.isdir(path_to_video):
        files=os.listdir(path_to_video)
        files.sort()
        frame_list=[_img_standardize(cv2.imread(os.path.join(path_to_video,file)),frame_height,frame_width)
                    for file in files]
    else:

        cap=cv2.VideoCapture(path_to_video)
        if not cap.isOpened():
            raise FileNotFoundError(f"can not find file {path_to_video}")

        while True:
            isOk,frame = cap.read()
            if not isOk:
                break
            frame = _img_standardize(frame, frame_height, frame_width)
            frame_list.append(frame)

    return frame_list

def clip_into_snippets(path_to_video, frame_height, frame_width, segment_count, is_random=False, augm_fn=None):

    '''

    :param path_to_video:
    :param frame_height:
    :param frame_width:
    :param segment_count:
    :return:  np array of shape [batch_size, segment_count, snippet_length, snippet_channels, height, width]
    '''

    frame_list = clip_to_frame_list(path_to_video,frame_height,frame_width)

    if len(frame_list)<segment_count:
        raise ValueError(f" error too few frames {len(frame_list)} < {segment_count}")

    snippets=[]
    frames_per_segment=len(frame_list)//segment_count

    for segment_idx in range(segment_count):
        if is_random:
            snippet = random.choice(frame_list[segment_idx * frames_per_segment: (segment_idx+1) * frames_per_segment])
        else:
            snippet = frame_list[segment_idx*frames_per_segment+frames_per_segment//2]

        if augm_fn is not None:
            snippet = random.choice(augm_fn)(snippet)
        snippets.append(snippet)

    snippets=np.stack(snippets,axis=0) #[segment_count,height,width,channels]
    snippets=np.transpose(snippets, (0,3,1,2)) #[segment_count,channels,height,width]
    snippets=np.expand_dims(snippets,axis=0) #[batch_size, segment_count,channels,height,width]
    snippets = np.expand_dims(snippets, axis=2) #[batch_size, segment_count, snippet_length, channels,height,width]
    snippets = snippets.astype(np.float32)
    return snippets


def noun_verb_from_path(pth_to_video):
    base = os.path.basename(pth_to_video)
    noun, verb = base.split('_')[:2]
    return noun, verb


def normalize_inputs_for_model(snippets,model):
    base=model.base_model
    if hasattr(base,'mean'):
        m = np.reshape(model.base_model.mean, (1, 1, 1, 3, 1, 1))
        s = np.reshape(model.base_model.std, (1, 1, 1, 3, 1, 1))
    else:
        if hasattr(model,'input_mean'):
            m = np.reshape(model.input_mean, (1, 1, 1, 3, 1, 1))
            s = np.reshape(model.input_std, (1, 1, 1, 1, 1, 1))
        else:
            raise NotImplementedError('unknown normalization params')

    if hasattr(base,'input_range'):
        if model.base_model.input_range==[0,1]:
            inputs = snippets / 255
        else:
            inputs = snippets / 255 - 0.5
    else:
        inputs = snippets

    inputs = torch.tensor((inputs - m) / s).float()
    return inputs


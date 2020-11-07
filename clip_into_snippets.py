import numpy as np
import cv2


def clip_into_snippets(path_to_video, frame_height, frame_width, segment_count):

    '''

    :param path_to_video:
    :param frame_height:
    :param frame_width:
    :param segment_count:
    :return:  np array of shape [batch_size, segment_count, snippet_length, snippet_channels, height, width]
    '''

    frame_list=[]
    cap=cv2.VideoCapture(path_to_video)
    if not cap.isOpened():
        raise FileNotFoundError(f"can not find file {path_to_video}")

    while True:
        isOk,frame = cap.read()
        if not isOk:
            break
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        frame_list.append(cv2.resize(frame,(frame_width,frame_height)))

    if len(frame_list)<segment_count:
        raise ValueError(f" error too few frames {len(frame_list)} < {segment_count}")

    snippets=[]
    frames_per_segment=len(frame_list)//segment_count

    for segment_idx in range(segment_count):
        snippet=frame_list[segment_idx*frames_per_segment+frames_per_segment//2]
        snippets.append(snippet)

    snippets=np.stack(snippets,axis=0) #[segment_count,height,width,channels]
    snippets=np.transpose(snippets, (0,3,1,2)) #[segment_count,channels,height,width]
    snippets=np.expand_dims(snippets,axis=0) #[batch_size, segment_count,channels,height,width]
    snippets = np.expand_dims(snippets, axis=2) #[batch_size, segment_count, snippet_length, channels,height,width]
    snippets = snippets.astype(np.float32)
    return snippets

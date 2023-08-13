import numpy as np
import lintel

'''
    video decode on the fly based on https://github.com/dukebw/lintel
'''
def _load_action_frame_nums_to_4darray(video, frame_nums, width, height):
    """Decodes a specific set of frames from `video` to a 4D numpy array.
    Args:
        video: Encoded video.
        dataset: Dataset meta-info, e.g., width and height.
        frame_nums: Indices of specific frame indices to decode, e.g.,
            [1, 10, 30, 35] will return four frames: the first, 10th, 30th and
            35 frames in `video`. Indices must be in strictly increasing order.
    Returns:
        A numpy array, loaded from the byte array returned by
        `lintel.loadvid_frame_nums`, containing the specified frames, decoded.
    """
    decoded_frames = lintel.loadvid_frame_nums(video,
                                               frame_nums=frame_nums,
                                               width=width,
                                               height=height)
    decoded_frames = np.frombuffer(decoded_frames, dtype=np.uint8)
    decoded_frames = np.reshape(
        decoded_frames,
        newshape=(len(frame_nums), height, width, 3))

    return decoded_frames
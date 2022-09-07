from abc import ABCMeta, abstractmethod

import numpy as np
import skimage
import cv2

from common.typing import NDArray

# Copied and modified from
# https://github.com/deepinsight/insightface/blob/master/recognition/arcface_mxnet/common/face_align.py

arcface_src = np.array([
  [38.2946, 51.6963],
  [73.5318, 51.5014],
  [56.0252, 71.7366],
  [41.5493, 92.3655],
  [70.7299, 92.2041] ], dtype=np.float32 )

arcface_src = np.expand_dims(arcface_src, axis=0)

class BaseImageCropper(metaclass=ABCMeta):
    """Base class for all model loader.
    All image alignment classes need to inherit this base class.
    """
    def __init__(self):
        pass

    @abstractmethod
    def crop_image_by_mat(self, image, landmarks):
        """Should be overridden by all subclasses.
        Used for online image cropping, input the original Mat, 
        and return the Mat obtained from the image cropping.
        """
        pass

class FaceCropper(BaseImageCropper):
    def __init__(self):
        super().__init__()
    
    def crop_image_by_mat(self, image: NDArray, landmarks: NDArray):
        assert(landmarks.shape == (5,2))
        _, _, channel = image.shape
        if channel != 3:
            print('Error input.')
        cropped_image = norm_crop(image, landmarks)
        return cropped_image

def norm_crop(img, landmark, image_size: int =112, mode: str = 'arcface'):
    M, _ = estimate_norm(landmark, image_size, mode)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue = 0.0)
    return warped

def estimate_norm(landmark, image_size: int =112, mode: str ='arcface'):
    assert landmark.shape == (5,2)
    transform = skimage.transform.SimilarityTransform()
    landmark_tran = np.insert(landmark, 2, values=np.ones(5), axis=1)
    min_M = []
    min_index = []
    min_error = float('inf')
    if mode == 'arcface':
        assert image_size == 112
        src = arcface_src
    else:
        raise NotImplementedError(f"{mode} is not supported")
    for i in np.arange(src.shape[0]):
        transform.estimate(landmark, src[i])
        M = transform.params[0:2, :]
        results = np.dot(M, landmark_tran.T)
        results = results.T
        error = np.sum(np.sqrt(np.sum((results - src[i])**2, axis=1)))
        #         print(error)
        if error < min_error:
            min_error = error
            min_M = M
            min_index = i
    return min_M, min_index

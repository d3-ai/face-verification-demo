import argparse
import os
import numpy as np

from typing import Any, List
from pathlib import Path
from PIL import Image

from facenet_pytorch import MTCNN
from utils.utils_face import FaceCropper

parser=argparse.ArgumentParser("Preprocessing face image for verfication.")
parser.add_argument("--dataset", type=str, required=True, choices=["usbcam", "CelebA"],help="dataset name for Centralized training")

def main():
    args = parser.parse_args()
    print(args)

    # configure path
    load_dir = Path("./data") / args.dataset.lower() / f"img_align_{args.dataset.lower()}"
    landmark_path = Path("./data") /args.dataset.lower() / f"list_landmarks_align_{args.dataset.lower()}.txt"
    
    if not os.path.isdir(load_dir):
        os.mkdir(load_dir)
    
    assert os.path.isdir(load_dir)
    
    if not os.path.isfile(landmark_path):
        # mtcnn from facenet-pytorch
        mtcnn = MTCNN(select_largest=False, post_process=False, device="cpu")

        # configure images
        num_images = sum(os.path.splitext(name)[1]=='.jpg' for name in os.listdir(load_dir))
        info: List[str] = ["" for _ in range(num_images)]

        for i in range(num_images):
            image_name: str = f"{i+1:06}.jpg"
            image_path = Path(load_dir) / image_name
            image = Image.open(image_path)

            # landmarks detection using MTCNN from facenet-pytorch
            _, _, _landmarks = mtcnn.detect(image, landmarks=True)
            assert _landmarks.size == 10
            info[i] = image_name + " " + " ".join([str(marker) for marker in _landmarks.reshape(-1)]) + "\n"
        
        with open(file=landmark_path, mode='x') as f:
            f.write(str(num_images)+"\n")
            f.write("lefteye_x lefteye_y righteye_x righteye_y nose_x nose_y leftmouth_x leftmouth_y rightmouth_x rightmouth_y\n")
            f.writelines(info)

    assert os.path.isfile(landmark_path)

    # Crop face images using the landmarks
    cropper = FaceCropper()

    with open(file=landmark_path, mode='r') as f:
        landmarks = f.read().split('\n')
    
    save_dir = Path("./data") / args.dataset.lower() / f"img_landmarks_align_{args.dataset.lower()}"
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    
    for line in landmarks[2:]:
        info: List[Any] = line.split()
        if len(info) == 0:
            continue
        face_name: str = info[0]
        face_landmark = np.array([float(x) for x in info[1:]], dtype=np.float32)
        face_landmark = face_landmark.reshape((5,2))

        face_path = load_dir / face_name
        face_nparray = np.array(Image.open(face_path))
        cropped_face = cropper.crop_image_by_mat(face_nparray, face_landmark)
        cropped_face_pil = Image.fromarray(cropped_face)
        save_path = save_dir / face_name
        cropped_face_pil.save(save_path)
        print(save_path)
        
if __name__ == "__main__":
    main()

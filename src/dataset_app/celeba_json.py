"""
following code is borrowed from LEAF
"""
import json
from pathlib import Path
import os

from torchvision.datasets.utils import download_file_from_google_drive
from typing import Dict, List, Tuple

TARGET_NAME = 'Eyeglasses'

def get_metadata() -> Tuple[List[str], List[str]]:
    """
    identities: List[str]
        identities[i] = '00000${i+1}.jpg 2880'
    attributes: List[str]
        attributes[0] = '202599'
        attributes[1] = '5_o_Clock_Shadow Arched_Eyebrows ...': attributes col
        attributes[i] = '00000${i-1}.jpg -1 1 1 ...': i>=2
    """
    f_identities = open('./data/celeba/identity_CelebA.txt', 'r')
    identities: List[str] = f_identities.read().split('\n')
    f_attributes = open('./data/celeba/list_attr_celeba.txt', 'r')
    attributes: List[str] = f_attributes.read().split('\n')
    return identities, attributes

def get_celebrities_and_images(identities: List[str]) -> Dict[str, List[str]]:
    """
    good_celebs: Dict[str, List[str]]
        key: celeb_id, value: image list
        good_celebs['2880'] = ['000001.jpg', ...]
        Each celeb contains at least 5 images.
    """
    all_celebs: Dict[str, List[str]] = {}

    for line in identities:
        info: List[str] = line.split()
        if len(info) < 2:
            continue
        image, celeb = info[0], info[1] # info[0]: '000001.jpg' info[1]: '2880'
        if celeb not in all_celebs:
            all_celebs[celeb] = []
        all_celebs[celeb].append(image)
    
    good_celebs: Dict[str, List[str]] = {c: all_celebs[c] for c in all_celebs if len(all_celebs[c]) == 30} 
    poor_celebs: Dict[str, List[str]] = {c: all_celebs[c] for c in all_celebs if len(all_celebs[c]) == 5} 
    return good_celebs, poor_celebs

def _get_celebrites_by_image(celebrities: Dict[str, List[str]]) -> Dict[str, str]:
    """
    good_images: Dict[str, str]
        key: image, value: celeb_id
        good_images['000001.jpg'] = '2880'
    """
    good_images = {}
    for c in celebrities:
        images = celebrities[c]
        for img in images:
            good_images[img] = c
    return good_images

def get_celebrities_and_target(celebrities: Dict[str, List[str]], attributes: List[str], attribute_name=TARGET_NAME)->Dict[str, List[int]]:
    col_name: str = attributes[1]
    col_idx: int = col_name.split().index(attribute_name)
    celeb_attributes: Dict[str, List[int]] = {}

    good_images = _get_celebrites_by_image(celebrities)

    for line in attributes[2:]:
        info = line.split()
        if len(info) == 0:
            continue
        image = info[0]
        if image not in good_images:
            continue
        celeb = good_images[image]

        # label conversion (1,-1) -> (1,0)
        att = int((int(info[1:][col_idx]) + 1) / 2)

        if celeb not in celeb_attributes:
            celeb_attributes[celeb] = []
        
        celeb_attributes[celeb].append(att)
    return celeb_attributes

def build_json_format(celebrities: Dict[str, List[str]], targets): 
    all_data = {}
    celeb_keys = [c for c in celebrities]
    num_samples = [len(celebrities[c]) for c in celeb_keys]
    data = {c: {'x': celebrities[c], 'y': targets[c]} for c in celebrities}

    all_data['users'] = celeb_keys
    all_data['num_samples'] = num_samples
    all_data['user_data'] = data
    return all_data


def write_json(json_data, train: bool = True):
    save_dir: str = os.path.join('./data/celeba/attrs/', TARGET_NAME)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    if train:
        file_path: str = Path(save_dir) / 'train_data.json'
    else:
        file_path: str = Path(save_dir) / 'test_data.json'
        
    print('writing {}'.format(file_path))
    with open(file_path, 'w') as outfile:
        json.dump(json_data, outfile)

def download():
    file_list = [
        # File ID                                      MD5 Hash                            Filename
        # ("0B7EVK8r0v71pZjFTYXZWM3FlRnM", "00d2c5bc6d35e252742224ab0c1e8fcb", "img_align_celeba.zip"),
        # ("0B7EVK8r0v71pbWNEUjJKdDQ3dGc","b6cd7e93bc7a96c2dc33f819aa3ac651", "img_align_celeba_png.7z"),
        # ("0B7EVK8r0v71peklHb0pGdDl6R28", "b6cd7e93bc7a96c2dc33f819aa3ac651", "img_celeba.7z"),
        ("0B7EVK8r0v71pblRyaVFSWGxPY0U", "75e246fa4810816ffd6ee81facbd244c", "list_attr_celeba.txt"),
        ("1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS", "32bd1bd63d3c78cd57e08160ec5ed1e2", "identity_CelebA.txt"),
        ("0B7EVK8r0v71pbThiMVRxWXZ4dU0", "00566efa6fedff7a56946cd1c10f1c16", "list_bbox_celeba.txt"),
        ("0B7EVK8r0v71pd0FJY3Blby1HUTQ", "cc24ecafdb5b50baae59b03474781f8c", "list_landmarks_align_celeba.txt"),
        # ("0B7EVK8r0v71pTzJIdlJWdHczRlU", "063ee6ddb681f96bc9ca28c6febb9d1a", "list_landmarks_celeba.txt"),
        ("0B7EVK8r0v71pY0NSMzRuSXJEVkk", "d32c9cbf5e040fd4025c592c306e6668", "list_eval_partition.txt"),
    ]
    for (file_id, md5, filename) in file_list:
        download_file_from_google_drive(file_id, "./data/celeba", filename, md5)

def verification_partition(train: bool =True, target: str = 'large'):
    # download()
    with open('./data/celeba/identity_CelebA.txt', 'r') as f:
        identities: List[str] = f.read().split('\n')
    
    celebrities, _ = get_celebrities_and_images(identities)
    print(len(celebrities))

    celeb_keys = [c for c in celebrities]
    if target == 'small':
        celeb_keys = celeb_keys[:10]
    elif target == 'medium':
        celeb_keys = celeb_keys[:100]
    elif target == 'large':
        celeb_keys = celeb_keys[10:1010]
    else:
        raise NotImplementedError(f"{target} is not supported")
    
    if train:
        data = {c: {'x': celebrities[c][:24], 'y': [i for _ in range(24)]} for i, c in enumerate(celeb_keys)}
        num_samples = [len(data[c]['x']) for c in celeb_keys]
    else:
        data = {c: {'x': celebrities[c][24:], 'y': [i for _ in range(6)]} for i, c in enumerate(celeb_keys)}
        num_samples = [len(data[c]['x']) for c in celeb_keys]
    all_data = {}
    all_data['users'] = celeb_keys
    all_data['num_samples'] = num_samples
    all_data['user_data'] = data

    save_dir: str = os.path.join('./data/celeba/identities/',target)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    if train:
        file_path: str = Path(save_dir) / 'train_data.json'
    else:
        file_path: str = Path(save_dir) / 'test_data.json'
        
    print('writing {}'.format(file_path))
    with open(file_path, 'w') as outfile:
        json.dump(all_data, outfile)
    

def main():
    download()
    identities, attributes = get_metadata()
    train_celebrities, test_celebrities = get_celebrities_and_images(identities)
    train_targets = get_celebrities_and_target(train_celebrities, attributes)
    train_json_data = build_json_format(train_celebrities, train_targets)
    test_targets = get_celebrities_and_target(test_celebrities, attributes)
    test_json_data = build_json_format(test_celebrities, test_targets)

    write_json(train_json_data, train=True)
    write_json(test_json_data, train=False)

if __name__ == "__main__":
    verification_partition(False)
    # main()

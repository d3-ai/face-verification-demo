# Face verification
## preprocessing
Crop face images using the detected five landmarks.
Face landmarks detection is performed by MTCNN from [face-pytorch](https://github.com/timesler/facenet-pytorch).
```bash
python ./face_verification/preprocess_face.py --dataset {DATASET_NAME}
```
On Raspberry Pi4B
```bash
export OMP_NUM_THREADS=1
python ./face_verification/preprocess_face.py --dataset {DATASET_NAME}
```
from typing import List
from facenet_pytorch import MTCNN
import torch
import cv2
from torchvision.transforms import transforms
from facefl.model import load_arcface_model
from facefl.dataset.preprocess_face import FaceCropper
# import matplotlib
# # matplotlib.use("TkAgg")
# import matplotlib.pyplot as plt
import Jetson.GPIO as GPIO
import time

# def decode_fourcc(v):
#         v = int(v)
#         return "".join([chr((v >> 8 * i) & 0xFF) for i in range(4)])
# def plot_face(target: int):
#     plt.close()
#     face_list = []
#     for i in range(10):
#         img_name = f"{i+1:06}.jpg"
#         img_path = f"./images/{img_name}"
#         img = Image.open(img_path)
#         img_array = np.asarray(img)
#         face_list.append(img_array)
#     on_img = cv2.imread("./images/on.jpg")
#     on_img = cv2.resize(on_img, dsize=(100,100))
#     off_img = cv2.imread("./images/off.jpg")
#     off_img = cv2.resize(off_img, dsize=(100,100))

#     # to put it into the upper left corner for example:
#     target = 0
#     fig = plt.figure(figsize=(5,5))
#     for row in range(2):
#         for col in range(5):
#             idx = row*5 + col
#             ax = fig.add_axes((0.025+col*0.2,0.6-row*0.3,0.15,0.15))
#             ax.imshow(face_list[idx])
#             ax.set_title(f"ID: {idx}")
#             ax.axis("off")
#             ax = fig.add_axes((0.075+col*0.2,0.53-row*0.3,0.05,0.05))
#             if idx == target:
#                 ax.imshow(cv2.cvtColor(on_img, cv2.COLOR_BGR2RGB))
#             else:
#                 ax.imshow(cv2.cvtColor(off_img, cv2.COLOR_BGR2RGB))
#             ax.axis("off")
#     plt.show(block=False)

def main(input_pins: List[int]):
    target = setup_gpio(input_pins=input_pins)
    transform = transforms.Compose([transforms.ToTensor()])
    net = load_arcface_model(name="GNResNet18", input_spec=(3,112,112), out_dims=1)
    net.eval()
    cropper = FaceCropper()
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
    camera.set(cv2.CAP_PROP_FPS, 24.)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    mtcnn = MTCNN(select_largest=False, post_process=False, device=device)

    thickness = 2
    scale = 1
    lineType = cv2.LINE_AA
    font = cv2.FONT_HERSHEY_SIMPLEX
    while True:
        _, frame = camera.read()
        frame_detect = frame.copy()
        boxes, _, landmarks = mtcnn.detect(frame_detect, landmarks=True)
        if boxes is not None:
            bbox = boxes.tolist()
            rec = (int(bbox[0][0]), int(bbox[0][1]), int(bbox[0][2] - bbox[0][0]), int(bbox[0][3] - bbox[0][1]))
            cv2.rectangle(frame, rec=rec, color = (255, 0, 0), thickness=thickness, lineType=lineType)
            cropped_face = cropper.crop_image_by_mat(frame_detect, landmarks[0])
            input_img = transform(cropped_face)
            input_img = input_img[None,:,:,:].to(device)
            score = float(net(input_img))
            if score >= 0.5:
                cv2.putText(frame, str(score)[:7], org=(int(bbox[0][0]), int(bbox[0][1] - 10)), fontFace=font, fontScale=scale, color=(0,255,0), thickness=thickness, lineType=lineType)
            else:
                cv2.putText(frame, str(score)[:7], org=(int(bbox[0][0]), int(bbox[0][1] - 10)), fontFace=font, fontScale=scale, color=(255,0,0), thickness=thickness, lineType=lineType)
    
        cv2.imshow('camera', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        for i in range(10):
            if GPIO.input(input_pins[i]) == GPIO.LOW:
                if target != i:
                    print(i)
                    weight = torch.load(f"./tmp/pi{i}/model.pth")
                    net.to("cpu")
                    net.load_state_dict(weight)
                    net.to(device)
                    target = i

    camera.release()
    cv2.destroyAllWindows()
    GPIO.cleanup()

def setup_gpio(input_pins: List[int]):
    assert len(input_pins) == 10
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(input_pins, GPIO.IN)
    values = []
    for pin in input_pins:
        print(pin)
        value = GPIO.input(pin)
        if value == GPIO.HIGH:
            values.append(1)
            print("pin : {}  value: {}".format(pin, 1))
        else:
            values.append(0)
            print("pin : {}  value: {}".format(pin, 0))
    assert sum(values) == 9
    return values.index(0)


if __name__ == "__main__":
    input_pins = [29, 31, 33, 35, 37, 22, 24, 26, 32, 36]
    main(input_pins=input_pins)


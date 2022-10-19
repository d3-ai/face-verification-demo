# miscs
import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class CentralizedCelebaVerification(Dataset):
    def __init__(self, target: str = "small", train: bool = True, transform=None) -> None:
        self.root = Path("./data/celeba")
        self.transform = transform
        if self.transform is None:
            self.transform = transforms.Compose([transforms.ToTensor()])

        if train:
            self.json_path = self.root / "identities" / target / "train_data.json"
        else:
            self.json_path = self.root / "identities" / target / "test_data.json"
        with open(self.json_path, "r") as f:
            self.json_data = json.load(f)

        self.num_samples = sum(self.json_data["num_samples"])

        self.data = {"x": [], "y": []}
        for _, data in self.json_data["user_data"].items():
            self.data["x"].extend(data["x"])
            self.data["y"].extend(data["y"])

    def __getitem__(self, index):
        img_path = self.root / "img_landmarks_align_celeba" / self.data["x"][index]
        img = Image.open(img_path)
        target = self.data["y"][index]

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return self.num_samples

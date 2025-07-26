import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, image_size=256, exts=(".png", ".jpg", ".jpeg"), random_flip=True):
        super().__init__()
        self.root_dir = root_dir
        self.paths = [
            os.path.join(root_dir, fname)
            for fname in sorted(os.listdir(root_dir))
            if fname.lower().endswith(exts)
        ]

        transform_list = [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
        if random_flip:
            transform_list.insert(0, transforms.RandomHorizontalFlip())

        self.transform = transforms.Compose(transform_list)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        return image

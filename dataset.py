import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from typing import List, Optional, Callable, Tuple


class ImageDataset(Dataset):
    def __init__(self, root: str, transform: Optional[Callable]) -> None:
        self.root: str = root
        self.transform: Optional[Callable] = transform
        self.images: List[str] = os.listdir(root)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[any]:
        image_path = os.path.join(self.root, self.images[idx])
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = transforms.PILToTensor()(image)
        return image

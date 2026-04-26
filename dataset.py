import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import config

class CaptchaDataset(Dataset):
    def __init__(self,data_dir):
        self.data_dir=data_dir
        self.files=[
            f for f in os.listdir(data_dir)
            if f.endswith('.png')
        ]
    def __len__(self):
        return len(self.files)
    def __getitem__(self,idx):
        filename=self.files[idx]
        label=filename.replace('.png','')
        image_path=os.path.join(self.data_dir,filename)
        image=Image.open(image_path).convert("RGB")
        return image,label
def build_dataloader(data_dir,batch_size,train_ratio):
    full_dataset=CaptchaDataset(data_dir)
    total=len(full_dataset)
    train_size=int(total*train_ratio)
    val_size=total-train_size
    train_dataset,val_dataset=random_split(full_dataset,[train_size,val_size])

    train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    val_loader=DataLoader(val_dataset,batch_size=batch_size,shuffle=False)
    return train_loader, val_loader
    

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
import matplotlib.pyplot as plt


IMAGE_SIZE = 256
BATCH_SIZE = 4
EPOCHS = 10
LR = 0.001

DATASET_NAME = "TsienDragon/figaro_hair_segmentation_1k"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


dataset = load_dataset(DATASET_NAME)

train_data = dataset["train"]
test_data = dataset["test"]

print(train_data.column_names)
print(train_data[0])


class HairDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset):
        self.data = hf_dataset

        self.img_transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor()
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]

        # Imagem original
        image = item["control_images"][0]

        # Máscara do cabelo
        mask = item["target_image"]

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        if not isinstance(mask, Image.Image):
            mask = Image.fromarray(mask)

        image = image.convert("RGB")
        mask = mask.convert("L")

        image = self.img_transform(image)
        mask = self.mask_transform(mask)

        mask = (mask > 0.5).float()

        return image, mask


train_dataset = HairDataset(train_data)
test_dataset = HairDataset(test_data)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.ReLU(inplace=True)
            )

        self.enc1 = block(3, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = block(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.middle = block(128, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = block(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = block(128, 64)

        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool1(e1)
        )

        m = self.middle(
            self.pool2(e2)
        )

        d2 = self.up2(m)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.out(d1)


model = UNet().to(DEVICE)

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR
)


from tqdm import tqdm

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    progress_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch+1}/{EPOCHS}"
    )

    for image, mask in progress_bar:

        image = image.to(DEVICE)
        mask = mask.to(DEVICE)

        pred = model(image)

        loss = criterion(pred, mask)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        progress_bar.set_postfix({
            "loss": loss.item()
        })

    avg_loss = total_loss / len(train_loader)

    print(f"\nEpoch Finalizada - Loss Média: {avg_loss:.4f}")

torch.save(model.state_dict(), "unet_hair_segmentation.pth")

print("Modelo salvo!")


model.eval()

image, real_mask = test_dataset[0]

with torch.no_grad():
    input_tensor = image.unsqueeze(0).to(DEVICE)

    pred = model(input_tensor)
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

pred = pred.squeeze().cpu().numpy()
image_np = image.permute(1, 2, 0).numpy()
real_mask_np = real_mask.squeeze().numpy()


plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("Imagem")
plt.imshow(image_np)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Máscara Real")
plt.imshow(real_mask_np, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Predição U-Net")
plt.imshow(pred, cmap="gray")
plt.axis("off")

plt.show()
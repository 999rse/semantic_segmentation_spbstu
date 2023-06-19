from imports import *
from make_dataset import unzipping, splitting, CustomImageDataset
from net import UNet
from train import train
from visualize import visualize_result

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

input_dir = 'data/raw'
output_dir = 'data/processed'

# Unzip dataset
unzipping(input_dir, output_dir)

# Split dataset on ti - train iamges, vi - validation images
ti, vi = splitting(output_dir)

# Create an instance for training, val data
print('Create an instance for training, val data')
training_data = CustomImageDataset(ti, ToTensor())
val_data = CustomImageDataset(vi, ToTensor())

train_dataloader = DataLoader(training_data, batch_size=4, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=4, shuffle=True)

# Create an instance of the class UNet
print('Create an instance of the class UNet')
model_unet = UNet(in_channels=3, out_channels=1).to(device)

# Declare the loss function and optimizer
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model_unet.parameters(), lr=0.0002)
print(f'Loss function: {loss_fn}, optimizer: {optimizer}')


# Traning zone
print('Traning start')
epochs = 1
for indep in tqdm(range(epochs)):
    train(train_dataloader, val_dataloader, model_unet, loss_fn, optimizer, epochs, device)
print("Training done!")

# Visualize the results on validation dataset
visualize_result(val_dataloader, model_unet, device)

from denoising.data import CAEDataset
from denoising.models import ConvAutoencoder
import torch
import torch.nn as nn
import numpy as np
import torchvision
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from scipy import interpolate


def imshow(img):
    img = img / 2 + 0.5
    plt.imshow(np.transpose(img, (1, 2, 0)))


def interpolate(img, scale):
    return nn.functional.interpolate(img, scale_factor=scale)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
transform = transforms.ToTensor()

mdt = False
n_epochs = 161
if mdt:
    var = 'mdt'
else:
    var = 'cs'
# Amend following code for my data
train_data = CAEDataset(
    region_dir=f'../a_mdt_data/HR_model_data/{var}_training_regions',
    quilt_dir=f'./quilting/DCGAN_{var}',
    mdt=mdt,
    transform=transforms.Compose([
        transforms.RandomRotation(90),
        transforms.RandomVerticalFlip,
        transforms.RandomHorizontalFlip,

    ])
    #write what I want - transforms.compose which allows to connect a bunch together
    # check whether the target and input transforms need to be fixed to be the same
)
test_data = CAEDataset(
    region_dir=f'../a_mdt_data/HR_model_data/{var}_testing_regions',
    quilt_dir=f'./quilting/DCGAN_{var}',
    mdt=mdt
    )

r = 2
a = 5
exp_epochs = [a * r**i for i in range(6)]

num_workers = 0
# batch_size = 64
batch_size = 128

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)

model = ConvAutoencoder()
print(model)
print(len(train_loader))
print(len(test_loader))
model.to(device)
criterion = nn.MSELoss(reduction='none')

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1, n_epochs+1):
    train_loss = 0.0
    for i, data in enumerate(train_loader):
        images = data[0].to(device)
        targets = data[1].to(device)
        outputs = model(images)
        land_mask = targets != 0
        # print('outputs shape: ', outputs.shape)
        # print('targets shape: ', targets.shape)
        # print('land mask shape: ', land_mask.shape)
        # print('output type: ', type(outputs))
        # print('target type: ', type(targets))
        # print('land mask type: ', type(land_mask))

        losses = []
        for scale in [0.25, 0.5, 1, 2, 4]:
            loss = criterion(interpolate(outputs, scale=scale), interpolate(targets, scale=scale))
            # print('scale: ', scale)
            land_mask = land_mask.float()
            mask = interpolate(land_mask, scale=scale)
            mask = mask.bool()
            # print('loss shape: ', loss.shape)
            # print('land mask shape: ', mask.shape)
            loss = loss * mask
            loss = loss.mean()
            losses.append(loss)
        optimizer.zero_grad()
        # loss = 0.3 * losses[0] etc
        loss = sum(losses)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*images.size(0)
        train_loss = train_loss/len(train_loader)
        print('[{}/{}] Epoch: {}/{} \tTraining Loss: {:.6f}'.format(
            i,
            len(train_loader),
            epoch,
            n_epochs,
            train_loss
            ))
        
        if epoch in exp_epochs:
            torch.save(model.state_dict(), f'models/multiscale_loss_{var}/{epoch}e_{var}_model_cdae.pth')

        #if epoch%10==0:

# torch.save(model.state_dict(), f'models/{n_epochs}e_{var}_model_cdae.pth')

inputs, targets = next(iter(test_loader))
inputs = inputs.to(device)
output = model(inputs)
inputs = inputs.cpu().numpy()
output = output.view(batch_size, 1, 128, 128)
output = output.detach().cpu().numpy()

# Consider whether it needs normalising?
# output = (output - output.min()) / (output.max() - output.min())
# output = output.clip(0, 1)
if mdt:
    # output = output.clip(-1.5, 1.5)
    vmin = -1.5
    vmax = 1.5
else:
    # output = output.clip(0, 2)
    vmin = 0
    vmax = 2
land_mask = np.array(targets != 0)
land_mask = np.logical_not(land_mask)
# output[land_mask] = -2

fig, axes = plt.subplots(nrows=3, ncols=10, sharex=True, sharey=True, figsize=(25,4))
for images, row in zip([inputs, output, targets], axes):
    images[land_mask] = -2
    for img, ax in zip(images, row):
        ax.imshow(np.squeeze(img), cmap='turbo', vmin=vmin, vmax=vmax)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

plt.show()
plt.close()


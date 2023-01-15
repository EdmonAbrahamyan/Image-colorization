import glob

import PIL
import torch
from matplotlib import pyplot as plt
from torchvision import transforms
import numpy as np
from tqdm import tqdm

from models import MainModel
from utils import lab_to_rgb, create_loss_meters, update_losses, log_results, visualize
from dataset import make_dataloaders
def train_model(model, train_dl, val_dl, epochs, display_every=200):
    data = next(iter(val_dl)) # getting a batch for visualizing the model output after fixed intrvals
    for e in range(epochs):
        loss_meter_dict = create_loss_meters() # function returing a dictionary of objects to
        i = 0                                  # log the losses of the complete network
        for data in tqdm(train_dl):
            model.setup_input(data)
            model.optimize()
            update_losses(model, loss_meter_dict, count=data['L'].size(0)) # function updating the log objects
            i += 1
            if i % display_every == 0:
                print(f"\nEpoch {e+1}/{epochs}")
                print(f"Iteration {i}/{len(train_dl)}")
                log_results(loss_meter_dict) # function to print out the losses
                visualize(model, data, save=False) # function di
              # splaying the model's outputs


if __name__ == '__main__':
    model = MainModel()
    paths = "Data set path"
    #paths = glob.glob("/home/edmon/PycharmProjects/pythonProject1/Image-colorization/Data/coco/images/train2017" + "/*.jpg")  # Grabbing all the image file names
    np.random.seed(123)
    paths_subset = np.random.choice(paths, 10_000, replace=False)  # choosing 1000 images randomly
    rand_idxs = np.random.permutation(10_000)
    train_idxs = rand_idxs[:8000]  # choosing the first 8000 as training set
    val_idxs = rand_idxs[8000:]  # choosing last 2000 as validation set
    train_paths = paths_subset[train_idxs]
    val_paths = paths_subset[val_idxs]

    train_dl = make_dataloaders(paths=train_paths, split='train')
    val_dl = make_dataloaders(paths=val_paths, split='val')

    train_model(model, train_dl, val_dl, 2)

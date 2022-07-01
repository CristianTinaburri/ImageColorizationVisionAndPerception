import os
import shutil
import torch
from skimage import color
import torchvision
import numpy as np
import matplotlib as plt


def copy_dataset_for_training(PATH_DATASET):
  PATH_TRAINING = PATH_DATASET + "/train"
  PATH_TESTING = PATH_DATASET + "/test"

  for root, dirs, files in os.walk(PATH_TRAINING):
   for file in files:
      path_file = os.path.join(root,file)
      shutil.copy2(path_file,'/content/training_images')


  for root, dirs, files in os.walk(PATH_TESTING):
    for file in files:
        path_file = os.path.join(root,file)
        shutil.copy2(path_file,'/content/testing_images')

def get_number_of_images():
  training = os.listdir('/content/training_images')
  number_files_training = len(training)
  print(number_files_training)

  test = os.listdir('/content/testing_images')
  number_files_testing = len(test)
  print(number_files_testing)

  return number_files_training, number_files_testing

def save_predictions(images_to_save=10):

  with torch.no_grad():
  
    # the number of the images you want to be predicted ( the same you want to be saved )
    number_of_images_grid = images_to_save

    # returns an array of random numbers of size of the variable number_of_images_grid
    image_inds = np.random.choice(len(test_set), number_of_images_grid, replace=False)

    # return a tensor with random elements of the size of the variable number_of_images_grid
    lab_batch = torch.stack([torch.cat([test_set[i][0], test_set[i][1]], dim=0) for i in image_inds])

    # ports lab_batch on the gpu
    lab_batch = lab_batch.to(device)

    # declaring the list that will contain the images to be predicted
    predicted_lab_batch = []

    # predicts the AB channels of the images using the trained model
    predicted_lab_batch = torch.cat([lab_batch[:, 0:1, :, :], model(lab_batch[:, 0:1, :, :])], dim=1)

    # ports lab_batch on the cpu
    lab_batch = lab_batch.cpu()

    # ports the predicted_lab_batch on the cpu
    predicted_lab_batch = predicted_lab_batch.cpu()

    # visualizing the images
    rgb_batch = []
    predicted_rgb_batch = []
    for i in range(lab_batch.size(0)):
        predicted_rgb_img = color.lab2rgb(np.transpose(predicted_lab_batch[i, :, :, :].numpy().astype('float64'), (1, 2, 0)))
        predicted_rgb_batch.append(torch.FloatTensor(np.transpose(predicted_rgb_img, (2, 0, 1))))

        rgb_img = color.lab2rgb(np.transpose(lab_batch[i, :, :, :].numpy().astype('float64'), (1, 2, 0)))
        rgb_batch.append(torch.FloatTensor(np.transpose(rgb_img, (2, 0, 1))))

    fig, ax = plt.subplots(figsize=(30, 30), nrows=1, ncols=2)
    ax[0].imshow(np.transpose(torchvision.utils.make_grid(torch.stack(predicted_rgb_batch), nrow=5).numpy(), (1, 2, 0)))
    ax[0].title.set_text('Predicted')
    ax[1].imshow(np.transpose(torchvision.utils.make_grid(torch.stack(rgb_batch), nrow=5).numpy(), (1, 2, 0)))
    ax[1].title.set_text('Reference')

    plt.savefig(MODEL_PATH_SAVE + MODEL_NAME + '_' + str(epoch) + '.png', bbox_inches='tight')
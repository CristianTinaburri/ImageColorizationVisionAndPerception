import os
import shutil

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
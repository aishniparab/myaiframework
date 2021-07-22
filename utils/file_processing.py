import torch
from torchvision import transforms
import imageio
import numpy as np

def image_file_to_array(filename,transform):
  """
  Takes an image path and returns numpy array
  Args:
    filename: Image filename
    dim_input: Flattened shape of image
  Returns:
    1 channel image
  """
  image = imageio.imread(filename, as_gray = True)  # misc.imread(filename)
  image = transform(image)
  h, w = image.size # 256 x 256
  image = np.asarray(image)
  image = image.reshape([h*w])
  image = image.astype(np.float32) / 255.0
  image = 1.0 - image
  image = image.reshape(1, h, w)
  image = torch.FloatTensor(image)
  image = image.reshape([h*w])
  return image 

def get_dirname(message, version):
  now = datetime.datetime.now()
  #'_'.join([now.year, now.month, now.day]+'_')
  date = '_'.join([str(val) for val in [now.year, now.month, now.day]])
  dir_name = '_'.join([message, date, version])
  return dir_name

def save_list_to_file(path, thelist):
  with open(path, 'w') as f:
    for item in thelist:
      f.write("%s\n" % item)

def make_export_dir(export_path):
  if not os.path.exists(path):
    os.makedirs(path)

def set_model_paths(export_path):
  best_model_path = os.path.join(export_dir, 'best_model.pth')
  last_model_path = os.path.join(export_dir, 'last_model.pth')

  return best_model_path, last_model_path


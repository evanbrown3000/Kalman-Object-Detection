import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from object_detection.utils import visualization_utils as viz_utils

def load_image_into_numpy_array(path, new_height, new_width, swap=True):
  """Load an image from file into a numpy array.

  Puts image into numpy array to feed into tensorflow graph.
  Note that by convention we put it into a numpy array with shape
  (height, width, channels), where channels=3 for RGB.

  Args:
    path: a file path.

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
  """
  image = Image.open(path)
  image = image.resize((new_width,new_height))

  image = np.array(image)
  image = image.reshape((new_height, new_width, 3))
  if swap == True:
    image = np.swapaxes(image,0,1)
  image = image.astype(np.uint8)
  return image

def plot_detections(image_np,
                    boxes,
                    classes,
                    scores,
                    category_index,
                    figsize=(12, 16),
                    image_save_name=None,
                    min_score_thresh=0.9):
  """Wrapper function to visualize detections.

  Args:
    image_np: uint8 numpy array with shape (img_height, img_width, 3)
    boxes: a numpy array of shape [N, 4]
    classes: a numpy array of shape [N]. Note that class indices are 1-based,
      and match the keys in the label map.
    scores: a numpy array of shape [N] or None.  If scores=None, then
      this function assumes that the boxes to be plotted are groundtruth
      boxes and plot all boxes as black with no classes or scores.
    category_index: a dict containing category dictionaries (each holding
      category index `id` and category name `name`) keyed by category indices.
    figsize: size for the figure.
    image_save_name: a name for the image file.
  """
  image_np_with_annotations = \
  viz_utils.visualize_boxes_and_labels_on_image_array(
      image = image_np,
      boxes = boxes,
      classes = classes,
      scores = scores,
      category_index = category_index,
      min_score_thresh = min_score_thresh,
      use_normalized_coordinates=True)
  if image_save_name:
    plt.imsave(image_save_name, image_np_with_annotations)
  else:
    return image_np_with_annotations

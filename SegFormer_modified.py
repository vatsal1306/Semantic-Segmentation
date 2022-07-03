

import os
import tempfile
import numpy as np
import tensorflow as tf
from PIL import Image
print(tf.__version__)

"""## Loading the model files"""

!pip install wget

#@title Downloading and extracting the model checkpoints

MODEL_NAME = "mobilenetv3_large_cityscapes_trainfine" #@param ["mobilenetv3_large_cityscapes_trainfine", "xception65_cityscapes_trainfine"] 

DOWNLOAD_URL_PREFIX = 'http://download.tensorflow.org/models/'
MODEL_URLS = {
    'mobilenetv3_large_cityscapes_trainfine':
        'deeplab_mnv3_large_cityscapes_trainfine_2019_11_15.tar.gz',
    'xception65_cityscapes_trainfine':
        'deeplabv3_cityscapes_train_2018_02_06.tar.gz',
}

MODEL_TAR = MODEL_URLS[MODEL_NAME]
MODEL_URL = DOWNLOAD_URL_PREFIX + MODEL_TAR

# Download
!wget -O {MODEL_TAR} {MODEL_URL}

# Extract
MODEL_FILE = !tar -zxvf {MODEL_TAR} --wildcards --no-anchored 'frozen_inference_graph.pb'
MODEL_FILE = MODEL_FILE[0].strip()
print('Frozen graph file path:', MODEL_FILE)

"""## Convert to TFLite"""

# Load the TensorFlow model
# The preprocessing and the post-processing steps should not be included in the TF Lite model graph 
# because some operations (ArgMax) might not support the delegates. 
# Insepct the graph using Netron https://lutzroeder.github.io/netron/
converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
    graph_def_file = MODEL_FILE, 
    input_arrays = ['sub_2'], # For the Xception model it needs to be `sub_7`, for MobileNet it would be `sub_2`
    output_arrays = ['ResizeBilinear_2']
)

# Optional: Perform the simplest optimization known as post-training dynamic range quantization.
# https://www.tensorflow.org/lite/performance/post_training_quantization#dynamic_range_quantization
# You can refer to the same document for other types of optimizations.
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert to TFLite Model
tflite_model = converter.convert()

_, tflite_path = tempfile.mkstemp('.tflite')
tflite_model_size = open(tflite_path, 'wb').write(tflite_model)
tf_model_size = os.path.getsize(MODEL_FILE)
print('TensorFlow Model is  {} bytes'.format(tf_model_size))
print('TFLite Model is      {} bytes'.format(tflite_model_size))
print('Post training dynamic range quantization saves {} bytes'.format(tf_model_size-tflite_model_size))

!ls -lh {tflite_path}

"""## Inference using TFLite model

### 1. Get Input Image Size
"""

# Load the model.
interpreter = tf.lite.Interpreter(model_path=tflite_path)

# Set model input.
input_details = interpreter.get_input_details()
interpreter.allocate_tensors()

# Get image size - converting from BHWC to WH # ([1,1025,2049,19]: Shape of ResizeBilinear_2 op)
input_size = input_details[0]['shape'][2], input_details[0]['shape'][1]
print(input_size)

"""# Original code"""

#@title 2. Provide a URL to your image to download
IMAGE_URL = 'https://www.outfrontmedia.ca/-/media/images/ofmcanada/media-type/street-furniture/street-furniture-creative-unlimited-banking.ashx' #@param {type:"string"}
!wget -O image {IMAGE_URL}

from PIL import Image
image = Image.open('image')
image

"""# My code"""

from PIL import Image
image = Image.open('/content/drive/MyDrive/Datasets/stuttgart_000131_000019_leftImg8bit.png')
image

"""#### Prepare the downloaded image for running inference"""

from PIL import ImageOps

old_size = image.size  # old_size is in (width, height) format
desired_ratio = input_size[0] / input_size[1]
old_ratio = old_size[0] / old_size[1]

if old_ratio < desired_ratio: # '<': cropping, '>': padding
    new_size = (old_size[0], int(old_size[0] / desired_ratio))
else:
    new_size = (int(old_size[1] * desired_ratio), old_size[1])

print(new_size, old_size)

# Cropping the original image to the desired aspect ratio
delta_w = new_size[0] - old_size[0]
delta_h = new_size[1] - old_size[1]
padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
cropped_image = ImageOps.expand(image, padding)
cropped_image

# Resize the cropped image to the desired model size
resized_image = cropped_image.convert('RGB').resize(input_size, Image.BILINEAR)

# Convert to a NumPy array, add a batch dimension, and normalize the image.
image_for_prediction = np.asarray(resized_image).astype(np.float32)
image_for_prediction = np.expand_dims(image_for_prediction, 0)
image_for_prediction = image_for_prediction / 127.5 - 1

"""Thanks to Khanh for helping to figure out the pre-processing and post-processing code.

### 3. Run Inference
"""

# Load the model.
interpreter = tf.lite.Interpreter(model_path=tflite_path)

# Invoke the interpreter to run inference.
interpreter.allocate_tensors()
interpreter.set_tensor(input_details[0]['index'], image_for_prediction)
interpreter.invoke()

# Retrieve the raw output map.
raw_prediction = interpreter.tensor(
    interpreter.get_output_details()[0]['index'])()

# Post-processing: convert raw output to segmentation output
## Method 1: argmax before resize - this is used in some frozen graph
# seg_map = np.squeeze(np.argmax(raw_prediction, axis=3)).astype(np.int8)
# seg_map = np.asarray(Image.fromarray(seg_map).resize(image.size, resample=Image.NEAREST))
## Method 2: resize then argmax - this is used in some other frozen graph and produce smoother output
width, height = cropped_image.size
seg_map = tf.argmax(tf.image.resize(raw_prediction, (height, width)), axis=3)
seg_map = tf.squeeze(seg_map).numpy().astype(np.int8)

"""The following code comes from [here](https://colab.research.google.com/github/tensorflow/models/blob/master/research/deeplab/deeplab_demo.ipynb). I took the colormap from [here](https://github.com/tensorflow/models/blob/master/research/deeplab/utils/get_dataset_colormap.py#L212). The label names come from [here](https://github.com/tensorflow/models/issues/6991#issue-454415742). """

from matplotlib import gridspec
from matplotlib import pyplot as plt

def create_cityscapes_label_colormap():
  """Creates a label colormap used in CITYSCAPES segmentation benchmark.
  Returns:
    A colormap for visualizing segmentation results.
  """
  colormap = np.zeros((256, 3), dtype=np.uint8)
  colormap[0] = [128, 64, 128]
  colormap[1] = [244, 35, 232]
  colormap[2] = [70, 70, 70]
  colormap[3] = [102, 102, 156]
  colormap[4] = [190, 153, 153]
  colormap[5] = [153, 153, 153]
  colormap[6] = [250, 170, 30]
  colormap[7] = [220, 220, 0]
  colormap[8] = [107, 142, 35]
  colormap[9] = [152, 251, 152]
  colormap[10] = [70, 130, 180]
  colormap[11] = [220, 20, 60]
  colormap[12] = [255, 0, 0]
  colormap[13] = [0, 0, 142]
  colormap[14] = [0, 0, 70]
  colormap[15] = [0, 60, 100]
  colormap[16] = [0, 80, 100]
  colormap[17] = [0, 0, 230]
  colormap[18] = [119, 11, 32]
  return colormap


def label_to_color_image(label):
  """Adds color defined by the dataset colormap to the label.

  Args:
    label: A 2D array with integer type, storing the segmentation label.

  Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the PASCAL color map.

  Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
  """
  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')

  colormap = create_cityscapes_label_colormap()

  if np.max(label) >= len(colormap):
    raise ValueError('label value too large.')

  return colormap[label]


'''def vis_segmentation(image, seg_map):
  """Visualizes input image, segmentation map and overlay view."""
  plt.figure(figsize=(15, 5))
  grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

  plt.subplot(grid_spec[0])
  plt.imshow(image)
  plt.axis('off')
  plt.title('input image')

  plt.subplot(grid_spec[1])
  seg_image = label_to_color_image(seg_map).astype(np.uint8)
  plt.imshow(seg_image)
  plt.axis('off')
  plt.title('segmentation map')

  plt.subplot(grid_spec[2])
  plt.imshow(image)
  plt.imshow(seg_image, alpha=0.7)
  plt.axis('off')
  plt.title('segmentation overlay')

  unique_labels = np.unique(seg_map)
  ax = plt.subplot(grid_spec[3])
  plt.imshow(
      FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
  ax.yaxis.tick_right()
  plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels])
  plt.xticks([], [])
  ax.tick_params(width=0.0)
  plt.grid('off')
  plt.show()
'''

LABEL_NAMES = np.asarray([
      'road',
      'sidewalk',
      'building',
      'wall',
      'fence',
      'pole',
      'traffic light',
      'traffic sign',
      'vegetation',
      'terrain',
      'sky',
      'person',
      'rider',
      'car',
      'truck',
      'bus',
      'train',
      'motorcycle',
      'bicycle',
])

FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)

image=cropped_image
plt.figure(figsize=(15, 5))
grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

plt.subplot(grid_spec[0])
plt.imshow(image)
plt.axis('off')
plt.title('input image')

plt.subplot(grid_spec[1])
seg_image = label_to_color_image(seg_map).astype(np.uint8)
plt.imshow(seg_image)
plt.axis('off')
plt.title('segmentation map')

plt.subplot(grid_spec[2])
plt.imshow(image)
plt.imshow(seg_image, alpha=0.6)
plt.axis('off')
plt.title('segmentation overlay')

unique_labels = np.unique(seg_map)
ax = plt.subplot(grid_spec[3])
plt.imshow(
    FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
ax.yaxis.tick_right()
plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels])
plt.xticks([], [])
ax.tick_params(width=0.0)
plt.grid('off')
plt.show()

"""# Segmented image"""

segmented_image=Image.fromarray(seg_image, 'RGB')
segmented_image

segmented_image.save("/content/drive/MyDrive/Datasets/pred.jpg")

type(segmented_image)

"""## Ground truth"""

gt=Image.open('/content/drive/MyDrive/Datasets/stuttgart_000131_000019_gtFine_color.png')
gt

type(gt)

print(gt.shape)

"""# MIoU"""

import cv2
from sklearn import metrics

truth=cv2.imread("/content/drive/MyDrive/Datasets/stuttgart_000131_000019_gtFine_color.png", 0).reshape(-1)
pred=cv2.imread("/content/drive/MyDrive/Datasets/pred.jpg", 0).reshape(-1)

truth.shape

type(truth)

type(pred)

pred.shape

truth = np.resize(truth, (2096128,))

A=truth
B=pred
print(A.shape)
print(B.shape)

len1=len(truth)
len2=len(pred)
if len1<len2:
  truth=np.append(truth, np.zeros((len2-len1)),axis=0)
elif len2<len1:
  pred=np.append(pred, np.zeros((len1-len2)),axis=0)
print(truth.shape)
print(pred.shape)
if (truth.shape != pred.shape):
  print("we have an error")
else:
  print("we r good")

if A.shape[0] < B.shape[0]:
    A = np.vstack((A, np.zeros((B.shape[0] - A.shape[0]))))
elif A.shape[0] > B.shape[0]:
    B = np.vstack((B, np.zeros((A.shape[0] - B.shape[0], A.shape[0])))) 

print(A.shape)
print(B.shape)

print("Accuracy", metrics.accuracy_score(truth, pred))

from keras.metrics import MeanIoU

pred1=pred/226

truth1=truth/194

truth

pred

mxresult=np.amax(pred)
mxresult

mxresult=np.amax(truth)
mxresult

num_classes=19
IOU_keras=MeanIoU(num_classes=num_classes)
IOU_keras.update_state(truth1, pred1)
print(IOU_keras.result().numpy())

image.paste(seg_image, mask=150)
image.show()

print(type(seg_image))

img1=Image.fromarray(image, 'RGB')

img=Image.fromarray(seg_image, 'RGB')

img.save("/content/drive/MyDrive/Datasets/output_img1.jpg")

from sklearn import metrics

print(metrics.accuracy_score(img1, img))

pred1=pred.flatten()
truth1=truth.flatten()

from sklearn.metrics import confusion_matrix

def compute_iou(y_pred, y_true):
     # ytrue, ypred is a flatten vector
    # y_pred = y_pred.flatten()
     #y_true = y_true.flatten()
     current = confusion_matrix(y_true, y_pred)
     # compute mean iou
     intersection = np.diag(current)
     ground_truth_set = current.sum(axis=1)
     predicted_set = current.sum(axis=0)
     union = ground_truth_set + predicted_set - intersection
     IoU = intersection / union.astype(np.float32)
     return np.mean(IoU)

compute_iou(pred1, truth1)

from keras import backend as K

def iou(y_true, y_pred, smooth = 100):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    union = K.sum(y_true,-1) + K.sum(y_pred,-1) - intersection
    #sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    iou_acc = (intersection + smooth) / (union + smooth)
    return iou_acc

iou(truth1, pred1)

"""# till here"""

vis_segmentation(cropped_image, seg_map)

from google.colab import drive
drive.mount('/content/drive')

seg_image

"""To try out a new model it's advisable to Factory Reset the runtime and then trying it.

# testing
"""

from keras import backend as K

!nvidia-smi

! cat /proc/cpuinfo

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('/content/file.csv')

sns.heatmap(df, cmap='coolwarm')
plt.xlabel('Predicted label')
plt.ylabel('True label')


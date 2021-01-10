import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import os
import io
import IPython.display
import numpy as np
import PIL.Image
from scipy.stats import truncnorm
import tensorflow_hub as hub
import math
import random

#module_path = 'https://tfhub.dev/deepmind/biggan-deep-256/1'
module_path = 'https://tfhub.dev/deepmind/biggan-deep-128/1'
# Load compressed models from tensorflow_hub
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'

tf.reset_default_graph()
print('Loading BigGAN module from:', module_path)
module = hub.Module(module_path)
inputs = {k: tf.placeholder(v.dtype, v.get_shape().as_list(), k)
          for k, v in module.get_input_info_dict().items()}
output = module(inputs)

print()
print('Inputs:\n', '\n'.join(
    '  {}: {}'.format(*kv) for kv in inputs.items()))
print()
print('Output:', output)



input_z = inputs['z']
input_y = inputs['y']
input_trunc = inputs['truncation']

dim_z = input_z.shape.as_list()[1]
print(dim_z)
vocab_size = input_y.shape.as_list()[1]

def truncated_z_sample(batch_size, truncation=1., seed=None):
  state = None if seed is None else np.random.RandomState(seed)
  values = truncnorm.rvs(-2, 2, size=(batch_size, dim_z), random_state=state)
  return truncation * values

def one_hot(index, vocab_size=vocab_size):
  index = np.asarray(index)
  if len(index.shape) == 0:
    index = np.asarray([index])
  assert len(index.shape) == 1
  num = index.shape[0]
  output = np.zeros((num, vocab_size), dtype=np.float32)
  output[np.arange(num), index] = 1
  return output

def one_hot_if_needed(label, vocab_size=vocab_size):
  label = np.asarray(label)
  if len(label.shape) <= 1:
    label = one_hot(label, vocab_size)
  assert len(label.shape) == 2
  return label

def sample(sess, noise, label, truncation=1., batch_size=8,
           vocab_size=vocab_size):
  noise = np.asarray(noise)
  label = np.asarray(label)
  num = noise.shape[0]
  if len(label.shape) == 0:
    label = np.asarray([label] * num)
  if label.shape[0] != num:
    raise ValueError('Got # noise samples ({}) != # label samples ({})'
                     .format(noise.shape[0], label.shape[0]))
  label = one_hot_if_needed(label, vocab_size)
  ims = []
  for batch_start in range(0, num, batch_size):
    s = slice(batch_start, min(num, batch_start + batch_size))
    feed_dict = {input_z: noise[s], input_y: label[s], input_trunc: truncation}
    ims.append(sess.run(output, feed_dict=feed_dict))
  ims = np.concatenate(ims, axis=0)
  assert ims.shape[0] == num
  ims = np.clip(((ims + 1) / 2.0) * 256, 0, 255)
  ims = np.uint8(ims)
  return ims

def save_sequence(images, format="png"):
  i = "0000"
  path = "test"
  while os.path.exists(f"{path}{i}"):
  		i = str(int(i) + 1).zfill(4)
  os.mkdir(f"{path}{i}")
  n = "00000"
  for img in images:
    while os.path.exists(f"{path}{i}/{n}.png"):
      n = str(int(n) + 1).zfill(4)
    img = np.asarray(img, dtype=np.uint8)
    PIL.Image.fromarray(img).save(f"{path}{i}/{n}.png", format)
  return f"{path}{i}"



def interpolate(A, B, num_interps):
  if A.shape != B.shape:
    raise ValueError('A and B must have the same shape to interpolate.')
  alphas = np.linspace(0, 1, num_interps)
  return np.array([(1-a)*A + a*B for a in alphas])


def sphere_sample(num_interps, mag=1):
    #define two orthogonal unit vec u, v
    u = np.random.normal(size = dim_z)
    u *= mag/np.linalg.norm(u)
    v = np.random.normal(size = dim_z)
    v -= v.dot(u) * u
    v *= mag/np.linalg.norm(v)

    alphas = np.linspace(0, math.pi * 2, num_interps)
    z = np.array([math.cos(t) * u + math.sin(t) * v for t in alphas])
    return z

def twist_z(angle, z):
  rot = np.identity(dim_z)
  axes = np.random.choice(dim_z, 2, replace=False)
  u = np.random.normal(size = dim_z)
  u /= np.linalg.norm(u)
  for i, x in enumerate(z):

    theta = (angle * x.dot(u)) / np.linalg.norm(x)
    rot[axes[0],axes[0]], rot[axes[1],axes[1]] = math.cos(theta), math.cos(theta)
    rot[axes[1],axes[0]], rot[axes[0],axes[1]] = math.sin(theta), -math.sin(theta)
    z[i] = np.matmul(rot,x)
  return z

def imgrid(imarray, cols=5, pad=1):
  if imarray.dtype != np.uint8:
    raise ValueError('imgrid input imarray must be uint8')
  pad = int(pad)
  assert pad >= 0
  cols = int(cols)
  assert cols >= 1
  N, H, W, C = imarray.shape
  rows = N // cols + int(N % cols != 0)
  batch_pad = rows * cols - N
  assert batch_pad >= 0
  post_pad = [batch_pad, pad, pad, 0]
  pad_arg = [[0, p] for p in post_pad]
  imarray = np.pad(imarray, pad_arg, 'constant', constant_values=255)
  H += pad
  W += pad
  grid = (imarray
          .reshape(rows, cols, H, W, C)
          .transpose(0, 2, 1, 3, 4)
          .reshape(rows*H, cols*W, C))
  if pad:
    grid = grid[:-pad, :-pad]
  return grid

def imshow(a, format='png', jpeg_fallback=True):
  a = np.asarray(a, dtype=np.uint8)
  data = io.BytesIO()
  #PIL.Image.fromarray(a).save(data, format)
  i = "0000"
  path = "test"
  while os.path.exists(f"{path}{i}.png"):
  	i = str(int(i) + 1).zfill(4)
  PIL.Image.fromarray(a).save(f"{path}{i}.png", format)
  im_data = data.getvalue()
  try:
    disp = IPython.display.display(IPython.display.Image(im_data))
  except IOError:
    if jpeg_fallback and format != 'jpeg':
      print(('Warning: image was too large to display in format "{}"; '
             'trying jpeg instead.').format(format))
      return imshow(a, format='jpeg')
    else:
      raise
  return disp

def create_movie(dir):
    filename = 'video.mp4'
    video = f"{dir}/{filename}.mov"
    outvideo = f"{dir}/{filename}.mp4"
    cmd = f'ffmpeg -y -i "{dir}/%04d.png" -c:v libx265 -preset slow -crf 17 -filter:v "format=yuv420p" -r 30 "{dir}/{filename}"'
    o = os.system(cmd)

def n_loops(sess, num_loops, num_frames, num_twists=1, mag=1, angle=1):
  for i in range(num_loops):
    z = sphere_sample(num_frames, mag=mag)
    for n in range(num_twists):
      z = twist_z(angle, z)
    y = random.randint(0,1000)
    ims = sample(sess, z, y, truncation=1., batch_size=1)
    path = save_sequence(ims)
    create_movie(path)
    print(f"Folder successfully created: {path}")


config = tf.compat.v1.ConfigProto(gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
config.gpu_options.allow_growth = True

initializer = tf.global_variables_initializer()
sess = tf.Session(config=config)
sess.run(initializer)

num_samples = 10
truncation = 1
noise_seed = 0
category = "109"

n_loops(sess, 1, 600, num_twists=4, mag=10, angle=4)
# S = truncated_z_sample(2, truncatnoise_seed)
# z = sphere_sample(600, mag=10)
# y = int(category.split(')')[0])

# ims = sample(sess, z, y, truncation=truncation, batch_size=1)
# path = save_sequence(ims)
# create_movie(path)
#import tensorflow as tf
import numpy as np
import math
import os
import PIL.Image


dim_z = 128

def sphere_sample(num_interps):
    #define two orthogonal unit vec u, v
    u = np.random.normal(size = dim_z)
    u = u / np.linalg.norm(u)
    v = np.random.normal(size = dim_z)
    v -= v.dot(u) * u
    v /= np.linalg.norm(v)

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
    print(np.linalg.norm(z[i] - np.matmul(rot,x)))
    z[i] = np.matmul(rot,x)
    print(z[i] - np.matmul(rot,x))
    print(z.shape)
    print(np.linalg.norm(z[i] - x))
  return z




def save_sequence1(images, format="png"):
    i = "0000"
    path = "test"
    while os.path.exists(f"{path}{i}"):
        i = str(int(i) + 1).zfill(4)
    os.mkdir(f"{path}{i}")
    for n, img in enumerate(images):
        img = np.asarray(img, dtype=np.uint8)
        PIL.Image.fromarray(img).save(f"{path}{i}/{str(n).zfill(5)}.png", format)

def save_sequence2(images, format="png"):
  i = "0000"
  path = "test"
  while os.path.exists(f"{path}{i}"):
        i = str(int(i) + 1).zfill(4)
  os.mkdir(f"{path}{i}")
  n = "00000"
  for img in images:
    while os.path.exists(f"{path}{i}/{n}"):
      n = str(int(n) + 1).zfill(4)
    img = np.asarray(img, dtype=np.uint8)
    PIL.Image.fromarray(img).save(f"{path}{i}/{n}.png", format)
  return f"{path}{i}"

def create_movie(dir):
    filename = 'video2.mp4'
    video = f"{dir}/{filename}.mov"
    outvideo = f"{dir}/{filename}.mp4"
    cmd = f'ffmpeg -y -i "{dir}/%04d.png" -c:v libx265 -preset slow -crf 17 -filter:v "format=yuv420p" -r 30 "{dir}/{filename}"'
    o = os.system(cmd)
images = np.random.uniform(0, 255, size=(5,128,128))

def n_loops(sess, z, y, num_loops, num_frames, mag):
  for i in range(n):
    z = sphere_sample(num_frames, mag=mag)
    y = randint(0,1000)
    ims = sample(sess, z, y, truncation=1., batch_size=1)
    path = save_sequence(ims)
    create_movie(path)
    print(f"Folder successfully created: {path}")

z_0 = sphere_sample(1)
z_1 = twist_z(40, z_0)
z_2 = z_0 - z_1
print(np.linalg.norm(z_2))

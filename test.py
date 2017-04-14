import tensorflow as tf
import numpy as np
from PIL import Image
from TPS_STN import TPS_STN

img = np.array(Image.open("original.png"))
out_size = list(img.shape)
shape = [1]+out_size+[1]

nx = 3
ny = 2

# Z ordering
v = np.array([
  [-1, -1],[0,-2],[1, -1],
  [-1, 1],[0,2],[1, 1]])

p = tf.constant(v.reshape([1, nx*ny, 2]), dtype=tf.float32)
t_img = tf.constant(img.reshape(shape), dtype=tf.float32)
t_img = TPS_STN(t_img, nx, ny, p, out_size)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  img1 = sess.run(t_img).reshape(out_size)
  img2 = np.hstack([img, img1])
  Image.fromarray(np.uint8(img2)).save("transformed.png") 

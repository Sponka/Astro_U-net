from __future__ import division
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import glob
from astropy.io import fits
import matplotlib.pyplot as plt
import os
import sys
from astropy.visualization import ZScaleInterval, LinearStretch, ImageNormalize
import math
import time

def lrelu(x):
    return tf.maximum(x * 0.2, x)

def upsample_and_concat(x1, x2, output_channels, in_channels, exp_time=None, exp=False):
    pool_size = 2
    deconv_filter = tf.Variable(tf.truncated_normal([pool_size, pool_size, output_channels, in_channels], stddev=0.02))
    deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2), strides=[1, pool_size, pool_size, 1])
    if not exp:
        deconv_output = tf.concat([deconv, x2], 3)
        deconv_output.set_shape([None, None, None, output_channels * 2])
    if exp:
        cons = tf.fill(tf.shape(deconv), exp_time)
        c = tf.slice(cons, [0, 0, 0, 0], [-1, -1, -1, 1])
        deconv_output = tf.concat([deconv, x2, c], 3)
        deconv_output.set_shape([None, None, None, output_channels * 2 + 1])
    return deconv_output

def network(input, e):
    conv1 = slim.conv2d(input, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_1')
    conv1 = slim.conv2d(conv1, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_2')
    pool1 = slim.max_pool2d(conv1, [2, 2], padding='SAME')

    conv2 = slim.conv2d(pool1, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv2_1')
    conv2 = slim.conv2d(conv2, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv2_2')
    pool2 = slim.max_pool2d(conv2, [2, 2], padding='SAME')

    conv3 = slim.conv2d(pool2, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3_1')
    conv3 = slim.conv2d(conv3, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3_2')
    pool3 = slim.max_pool2d(conv3, [2, 2], padding='SAME')

    conv4 = slim.conv2d(pool3, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv4_1')
    conv4 = slim.conv2d(conv4, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv4_2')
    pool4 = slim.max_pool2d(conv4, [2, 2], padding='SAME')

    conv5 = slim.conv2d(pool4, 512, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv5_1')
    conv5 = slim.conv2d(conv5, 512, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv5_2')

    up6 = upsample_and_concat(conv5, conv4, 256, 512, exp_time=e, exp=True)
    conv6 = slim.conv2d(up6, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_1')
    conv6 = slim.conv2d(conv6, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_2')

    up7 = upsample_and_concat(conv6, conv3, 128, 256)
    conv7 = slim.conv2d(up7, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_1')
    conv7 = slim.conv2d(conv7, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_2')

    up8 = upsample_and_concat(conv7, conv2, 64, 128)
    conv8 = slim.conv2d(up8, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_1')
    conv8 = slim.conv2d(conv8, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_2')

    up9 = upsample_and_concat(conv8, conv1, 32, 64)
    conv9 = slim.conv2d(up9, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_1')
    conv9 = slim.conv2d(conv9, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_2')

    conv10 = slim.conv2d(conv9, 1, [1, 1], rate=1, activation_fn=None, scope='g_conv10')
    return conv10

def poisson_noise(name,ratio = 2,exp = None, xx = None, yy = None, ps = None, ron = 3, dk = 7, sv_name = None, path = None, patch_noise = False, ret = True, save = False):
    img_data = name #e/sec
    #print(img_data.shape)
    if patch_noise:
        img_data = img_data[xx:xx + ps, yy:yy + ps]
    width, height = img_data.shape[0:2]
    exp_time = exp 
    img = img_data * exp_time   # e
    DN = np.random.normal(0, np.sqrt(dk*exp_time/((60*60)*ratio)), (width, height))    
    RON = np.random.normal(0, ron, (width, height))

    SN = np.random.poisson(np.abs(img/ratio))   # this line is different     
    noise_img = (SN + RON + DN)/(exp_time/ratio)
    noise_img = np.where(img_data == 0.00000000e+00, 0.00000000e+00  , noise_img)
    if save:
        save_fits(noise_img, sv_name, path)
    if ret:    
        return noise_img

def fits_read(name):
    """reads exposure time of image
       if information is not in header raise exception"""
    f = fits.open(name)
    if "EXPTIME" in f[0].header:
        e = f[0].header["EXPTIME"]
        return e
    else:
        raise Exception('Exposure time was not found in header. Remove image ' + str(name) + ' from the dataset' )

def valid_img(path,ratio, sv_path, create_noise = True):
    """test network during training
       ratio - exposure time ratio
       path - path to original (image with noise)
       sv_path - where you wanna save your image"""    
    data = np.sort(glob.glob(path +"*.fits"))
    print(data)
    tt = time.time()
    for i in range(len(data)):
        t = time.time()
        one_image(data[i],ratio, sv_path, create_noise = create_noise)
        print(time.time()-t)
    print("Validation done")
    print(time.time()-tt)

def one_image(name, ratio, sv_path, create_noise = True, save=True, save_png = False):
    """iterate whole image, predict output and make one big image
        name - name of image 
               if create_noise = True - it is name (path to) of original image without noie
               if create_noise = False - it is name (path to) of noisy image
        sv_path - where you wanna save your image
        if save = True - save fits image
        if save_png = true - save png image

	ratio - int number how much we wanna enhance the image
	ps - is size of image which goes to network
	     we use 256 bc of memory 
	ex_time = exposure time of the image, read from header of fits file
	step - int number, how much we move on the image
	        0 < step < ps, we used step < ps bc if step = ps checkerboard pattern is created
                ps % step = 0 
	W, H - int numbers, size of the img
	w_diff/h_diff - float numbers, used to create boarder around the image if W%256!=0 or H%256!=0"""
    print(name)
    print(ex_time)

    ps = 256
    if create_noise:
        print('noise')
        ex_time = fits_read(name)
        input_image = poisson_noise(name, exp_time=ex_time, ratio=ratio, ret=True)
    if not create_noise:
        input_image = fits.getdata(name, ext=0)
        input_image = np.where(input_image < 0.00000000e+00, 0.00000000e+00  , input_image)
    header = fits.getheader(name) 
    step = 32  # you can change your step, max_step = ps (256)
    if ps %step != 0:
        raise Exception('ps % step should be 0. The value of ps % step was: {}'.format(ps%step))
    img = input_image
    W, H = img.shape[0: 2]
    #this part create a border around image, to eliminate any checkerboard pattern
    w_diff = (ps - (W % 256))
    w_c = ps + int(np.ceil(w_diff/2))
    w_f = ps + int(np.floor(w_diff/2))
    h_diff = (ps - (H % 256))
    h_c = ps + int(np.ceil(h_diff/2))
    h_f = ps + int(np.floor(h_diff/2))
    w = W + w_c + w_f
    h = H + h_c + h_f
   
    image = np.zeros((w, h, 1))
    image[w_c: w_c + W, h_c: h_c + H, 0] = img
    output = np.zeros((w, h))
    ratio = 2
    for i in range(0, w, step):
        for j in range(0, h,  step):
            in_patch = image[i: i+ps, j: j+ps]
            in_patch = np.expand_dims(in_patch,  axis=0)
            new_image = sess.run(out_image, feed_dict={in_image: in_patch, ex: ratio})
            output[i: i+ps, j: j+ps] += new_image[0, :, :, 0]
    output = (output/((ps/step)**2))[w_c: w_c + W, h_c: h_c + H] *ex_time # to return image in electrons



    if save:
        save_name = sv_path + name.split('/')[-1][:-5] # save name
        fits.writeto(save_name + '.fits', output, header)
    if save_png:
        norm = ImageNormalize(input_image, interval=ZScaleInterval(), stretch=LinearStretch()) #for png images normalization is important! you can play with it
        plt.imshow(output, cmap='Greys_r', origin='lower', norm=norm)
        plt.axis('off')
        plt.savefig(save_name + ".png", dpi=600, bbox_inches='tight', pad_inches=0)







sess = tf.Session()
in_image = tf.placeholder(tf.float32, [None, None, None, 1])
gt_image = tf.placeholder(tf.float32, [None, None, None, 1])
ex = tf.placeholder(tf.float32, name="TIME")
out_image = network(in_image, ex)
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())


ratio = 2
checkpoint_dir =   xxx
sv_path =          xxx  # path where you saved the image
saver.restore(sess, checkpoint_dir + 'model5000.ckpt')
print('working')
path =             xxx # path for your input images, if you work with images without noise, set  create_noise = True
valid_img(path,ratio, sv_path, create_noise = False)


# valid_img(path,ratio, sv_path, create_noise = True)
  


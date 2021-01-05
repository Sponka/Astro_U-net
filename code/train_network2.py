from __future__ import division
import time
import scipy.io
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import glob
from astropy.io import fits
import matplotlib.pyplot as plt
import os
import sys
from astropy.stats import biweight_location
from astropy.stats import mad_std
from astropy.stats import sigma_clipped_stats
from datetime import datetime
from datetime import timedelta
import random

# folders names
#-------------------------------
folder_num = str('train_network2')
# create_image_noise = False
gt_dir = './data_paper/out_data/'             # ground truth
checkpoint_dir = './' + folder_num + '/'
result_dir = './' + folder_num + '/'   # results folder
test_net = './' + folder_num + '/test_net/'
name_txt_file = result_dir + "train"  + ".txt"
name_txt_file_mean = result_dir + "test_mean" + ".txt"
name_txt_file_eval = result_dir + "eval_loss"  + ".txt"
name_txt_file_epoch = result_dir + "train_epoch" + ".txt"
#-------------------------------


# other parameters
#-------------------------------
ps = 256  # patch size for training
save_freq = 500
ratio = 2
ron = 3 #e
dk = 7 #e/hr/pix;
#-------------------------------



if not os.path.isdir(checkpoint_dir):
                os.makedirs(checkpoint_dir )

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

def upsample_nn_and_concat(x1, x2, output_channels, in_channels, exp_time=None, exp=False):
    up_sample = tf.image.resize_bilinear(x1, (tf.shape(x2)[1], tf.shape(x2)[2]))
    conv_up = slim.conv2d(up_sample, output_channels, [3, 3], rate=1, activation_fn=None)
    if not exp:
        deconv_output = tf.concat([conv_up, x2], 3)
        deconv_output.set_shape([None, None, None, output_channels * 2])
    if exp:
        cons = tf.fill(tf.shape(conv_up), exp_time)
        c = tf.slice(cons, [0, 0, 0, 0], [-1, -1, -1, 1])
        deconv_output = tf.concat([conv_up, x2, c], 3)
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


def save_fits(image, name, path):
    hdu = fits.PrimaryHDU(image)
    hdul = fits.HDUList([hdu])
    hdul.writeto(path + name + '.fits')


def squeeze_img(array, fil, ps = 256):
    """change dimension of network output"""
    out = np.squeeze(array, axis = 0)
    o = np.zeros((ps, ps, 1))
    o[:,:,0] = out[:,:,fil]
    out = np.squeeze(o, axis = 2)
    return out


def text_write(file, epoch, cnt, loss,mean_loss, time, name):
    """save parameters into text file"""
    file = open(file, "a")
    file.write(str(epoch) + "\t" + str(cnt) + "\t" + str(loss) + "\t" + str(mean_loss) + "\t" + str(time) + "\t" + str(name) + "\n")
    file.close


def black_level(arr, max_num, level=0.1):
    """Prevent to have an image with more than some percentage of zeroes as input
       level - percentage <0,1>; 0.1/10% default"""
    arr = list(np.hstack(arr))
    per = arr.count(0.00000000e+00)/len(arr)
    if max_num > 10:
        level = 0.3
    if per < level or max_num > 15:
        return True
    else:
        return False



def poisson_noise(name,ratio = 2,exp = None, xx = None, yy = None, ps = None, ron = 3, dk = 7, sv_name = None, path = None, patch_noise = False, ret = True, save = False):
    img_data = name #e/sec
    if patch_noise:
        img_data = img_data[xx:xx + ps, yy:yy + ps]
    width, height = img_data.shape[0:2]
    exp_time = exp 
    img = img_data * exp_time   # e
    DN = np.random.normal(0, np.sqrt(dk*exp_time/((60*60)*ratio)), (width, height))    
    RON = np.random.normal(0, ron, (width, height))

    SN = np.random.poisson(img/ratio)   # this line is different     
    noise_img = (SN + RON + DN)/(exp_time/ratio)
    noise_img = np.where(img_data == 0.00000000e+00, 0.00000000e+00  , noise_img)
    if save:
        save_fits(noise_img, sv_name, path)
    if ret:
        return noise_img




def text_write_data(file, l):
    """save parameters into text file"""
    file = open(file, "a")
    for name in l:
        file.write(str(name) + "\n")
    file.close



def out_in_image(name):
    ps = 256
    out = fits.getdata(name, ext = 0)
    f = fits.open(name)
    if "EXPTIME" in f[0].header:
        ex_time = f[0].header["EXPTIME"]
        print(name, ' exp in the header: ', ex_time)
    else:
        ex_time = float(int(name[-10:-5]))
        print(name, ' exp in the name: ', ex_time)
    if  len(out.shape) == 1:
             out = fits.open(name)[3].data
             print('fits in  extension')
    H, W = out.shape[0], out.shape[1]
    zero_level = False
    max_num = 0
    while not zero_level:
            xx = np.random.randint(0, H - ps)
            yy = np.random.randint(0, W - ps)
            arr = out[xx:xx + ps, yy:yy + ps]
            zero_level = black_level(arr, max_num)
            max_num += 1
    # ground truth
    out = np.where(out < 0.00000000e+00, 0.00000000e+00  , out)
    out_img = np.zeros((ps, ps, 1))
    out_img[:, :, 0] = out[xx:xx + ps, yy:yy + ps]
    gt_patch = np.expand_dims(out_img, axis=0)

    # input image
    in_imag =  poisson_noise(out[xx:xx + ps, yy:yy + ps], exp = ex_time)
    in_img = np.zeros((ps, ps, 1))
    in_img[:, :, 0] = in_imag
    in_patch = np.expand_dims(in_img, axis=0)
    return  gt_patch,in_patch, out
    
def read_list(name_list):
    all_names = []
    data = open(str(name_list), 'r') 
    lines = data.read().split('\n')
    data.close()
    for line in lines:
        all_names.append(str(line))
    all_names = all_names[:-1]
    print(len(all_names))
    print(all_names)
    return all_names


sess = tf.Session()


in_image = tf.placeholder(tf.float32, [None, None, None, 1])
gt_image = tf.placeholder(tf.float32, [None, None, None, 1])
ex = tf.placeholder(tf.float32, name="TIME")
out_image = network(in_image, ex)


G_loss = tf.reduce_mean(tf.abs(out_image - gt_image))
t_vars = tf.trainable_variables()


lr = tf.placeholder(tf.float32)
G_opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(G_loss, var_list=[var for var in tf.trainable_variables()])
saver = tf.train.Saver(max_to_keep = 1000)
sess.run(tf.global_variables_initializer())
ckpt = tf.train.get_checkpoint_state(checkpoint_dir)




g_loss = []

train_data = read_list('./train_name.txt')
eval_data = read_list('./eval_name.txt')
print(len(train_data), len(eval_data))



g_loss = []
eval_loss = []
data_list = glob.glob(gt_dir + "*.fits")
#if ckpt:
 #   print('loaded ' + ckpt.model_checkpoint_path)
 #   saver.restore(sess, './train_network2/model4000.ckpt')

learning_rate = 1e-4
for epoch in range(0, 5001):
    num = 0
    if os.path.isdir('./train_network2/%04d' % epoch):
        continue
    cnt = 0
    if epoch > 2000:
        learning_rate = 1e-5
    epoch_loss = []
    epoch_eval_loss = []
    for name in np.random.permutation(train_data):
        st = time.time()
        num += 1
        print(name)
        out = fits.open(name, ext=0)
        if "EXPTIME" in out[0].header:
            ex_time = out[0].header['EXPTIME']
            print(name, ' exp in the header: ', ex_time)

        if "EXPTIME" not in out[0].header:
            ex_time =  float(int(name[-10:-5])) 
            print(name, ' exp in the name: ', ex_time)

        out = fits.getdata(name)
        if  len(out.shape) == 1:
             out = fits.open(name)[3].data
             print('fits in  extension')
        H, W = out.shape[0], out.shape[1]
       
        zero_level = False
        max_num = 0
        while not zero_level:
            xx = np.random.randint(0, H - ps)
            yy = np.random.randint(0, W - ps)
            arr = out[xx:xx + ps, yy:yy + ps]
            zero_level = black_level(arr, max_num)
            max_num += 1
        
        out = np.where(out < 0.00000000e+00, 0.00000000e+00  , out)
        # ground truth
        out_img = np.zeros((ps, ps, 1))
        out_img[:, :, 0] = out[xx:xx + ps, yy:yy + ps]
        gt_patch = np.expand_dims(out_img, axis=0)


        for r in range(2,6):
            rat = r      
            # input image
            in_imag = poisson_noise(out[xx:xx + ps, yy:yy + ps], exp= ex_time, ratio=rat, ps=256, ron=3, dk=7)
            in_img = np.zeros((ps, ps, 1))
            in_img[:, :, 0] = in_imag
            in_patch = np.expand_dims(in_img, axis=0)           
            _, G_current, output = sess.run([G_opt, G_loss, out_image],
                                        feed_dict={in_image: in_patch, gt_image: gt_patch, lr: learning_rate, ex: rat})
            
            epoch_loss.append(G_current)
            g_loss.append(G_current)
            print("%d %d Loss=%.3f Time=%.3f Ratio=%.1f" % (epoch, cnt, np.mean(g_loss), time.time() - st, rat))
        cnt += 1
        if cnt % 50 == 0:
            text_write(name_txt_file, epoch, cnt, G_current, np.mean(g_loss), time.time() - st, name.split('/')[-1])

        if  cnt % 40 == 0:
            name_eval =  random.choice(eval_data)
            out_eval, in_eval, _ = out_in_image(name_eval)
            ratio = 2
            eval_current, output_eval = sess.run([G_loss, out_image],
                                        feed_dict={in_image: in_eval, gt_image: out_eval, ex: ratio})
            eval_loss.append(eval_current)
            epoch_eval_loss.append(eval_current)
            text_write(name_txt_file_eval, epoch, cnt, eval_current, np.mean(epoch_eval_loss), np.mean(eval_loss), name.split('/')[-1])
        if epoch % save_freq == 0 and cnt % 100 == 0:
            if not os.path.isdir(result_dir + '%04d' % epoch):
                os.makedirs(result_dir + '%04d' % epoch)
            sv_name = "/" + name.split('/')[-1][:-5]
            output = squeeze_img(output, 0)
            save_fits(out[xx:xx + ps, yy:yy + ps], sv_name, result_dir + '%04d' % epoch)
            save_fits(output, sv_name + "_output", result_dir + '%04d' % epoch)
            save_fits(in_imag, sv_name + "_noise", result_dir + '%04d' % epoch)       
    if epoch >= 1000 and epoch % (save_freq) == 0:
        print('model%04d.ckpt' % epoch)
        saver.save(sess, checkpoint_dir + 'model%04d.ckpt' % epoch)
    saver.save(sess, checkpoint_dir + 'model.ckpt')
    text_write(name_txt_file_epoch, epoch, cnt, np.mean(epoch_loss),  np.mean(g_loss), '0', name.split('/')[-1])
    



































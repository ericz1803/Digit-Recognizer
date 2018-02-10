import numpy as np
import tensorflow as tf
import math
import cv2
import os
import matplotlib.pyplot as plt
#get user image
import get_number_pixels

global cwd
cwd=os.getcwd()

def get_img_and_preprocess(show=True):
    get_number_pixels.get_pixels()
    #read in image
    pic=cv2.imread(os.path.join(cwd,"screenshot.png"))
    os.remove(os.path.join(cwd,"screenshot.png"))
    #convert it into grayscale
    pic=cv2.cvtColor(pic,cv2.COLOR_BGR2GRAY)
    #resize to 28x28 (mnist size) using opencv anti-aliasing
    pic=cv2.resize(pic, (28,28), fx=0.5, fy=0.5, interpolation = cv2.INTER_AREA)
    if show:
        plt.imshow(pic, cmap='gray')
        plt.title('Image after anti-aliasing and changing the size to 28x28px.')
        plt.show()
    #preprocess(translate it to center of mass)
    intensity_sum=np.sum(pic)
    pic=np.array(pic)
    x_sum=0
    y_sum=0
    for x in range(pic.shape[0]):
        for y in range(pic.shape[1]):
            x_sum+=x*pic[x,y]
            y_sum+=y*pic[x,y]
    x_sum=np.round(x_sum/intensity_sum)
    y_sum=np.round(y_sum/intensity_sum)

    #M is translation matrix
    M = np.float32([[1,0,14-y_sum],[0,1,14-x_sum]])
    pic = cv2.warpAffine(pic, M, (28, 28))
    if show:
        plt.imshow(pic, cmap='gray')
        plt.title('After Translation')
        plt.show()
    #flatten it and normalize values
    pic=pic.reshape(1,784)/255

    #returns 28x28 numpy array
    return pic

nodes_hl1=1000
nodes_hl2=500
nodes_hl3=100

output_size=10



#create variables

weights_input_hl1=tf.get_variable('weights_input_hl1', dtype=tf.float32, 
  initializer=tf.truncated_normal([784, nodes_hl1], dtype=tf.float32, stddev=np.sqrt(2/784)))
biases_hl1=tf.get_variable('biases_hl1', [nodes_hl1], dtype=tf.float32, 
  initializer=tf.zeros_initializer)

weights_hl1_hl2=tf.get_variable('weights_hl1_hl2', dtype=tf.float32, 
  initializer=tf.truncated_normal([nodes_hl1, nodes_hl2], dtype=tf.float32, stddev=np.sqrt(2/nodes_hl1)))
biases_hl2=tf.get_variable('biases_hl2', [nodes_hl2], dtype=tf.float32, 
  initializer=tf.zeros_initializer)

weights_hl2_hl3=tf.get_variable('weights_hl2_hl3', dtype=tf.float32, 
  initializer=tf.truncated_normal([nodes_hl2, nodes_hl3], dtype=tf.float32, stddev=np.sqrt(2/nodes_hl2)))
biases_hl3=tf.get_variable('biases_hl3', [nodes_hl3], dtype=tf.float32, 
  initializer=tf.zeros_initializer)

weights_hl3_output=tf.get_variable('weights_hl3_output', dtype=tf.float32, 
  initializer=tf.truncated_normal([nodes_hl3, output_size], dtype=tf.float32, stddev=np.sqrt(2/nodes_hl3)))

#create saver to restore values
saver=tf.train.Saver(max_to_keep=1)

def forward_pass(x):
    l1=tf.add(tf.matmul(x, weights_input_hl1), biases_hl1)
    l1=tf.nn.elu(l1)
    l2=tf.add(tf.matmul(l1, weights_hl1_hl2), biases_hl2)
    l2=tf.nn.elu(l2)
    l3=tf.add(tf.matmul(l2, weights_hl2_hl3), biases_hl3)
    l3=tf.nn.elu(l3)
    output_layer=tf.matmul(l3, weights_hl3_output)
    return tf.nn.softmax(output_layer)

features=tf.placeholder(tf.float32,shape=[None,784])
prediction=tf.argmax(forward_pass(features), 1)

with tf.Session() as sess:
    #initialize variables
    sess.run(tf.global_variables_initializer())
    #restore weights from save file trained on the mnist dataset
    try:
        saver.restore(sess, os.path.join(cwd,"weights/model.ckpt"))
        print("Model restored.")
    except:
        print("Error: No save file found")
    
    enter_another_img='yes'
    while enter_another_img.lower()[0]=='y':
        pic=get_img_and_preprocess(show=False)
        print("Prediction: {}".format(sess.run(prediction, feed_dict={features: pic})[0]))
        enter_another_img=input("Draw another number? (y/n)")


# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 17:42:20 2018

@author: LEE
"""

import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split

#data-set read
movie = pd.read_csv("./data/movie.csv", encoding ='CP949')

#data set의 열을 분리(slice)

#label 데이터를 제외한 영화관련 데이터
movie_data = movie.iloc[:,:-1]

#맨 마지막 열인 label 데이터
label_data = movie.iloc[:,-1:]

#전체 data set중 30%를 test데이터, 70%는 학습에 사용하기 위해 임의로 분할 
train_x, test_x, train_y, test_y = train_test_split(movie_data, label_data, test_size=0.3, random_state=42, stratify=label_data)

#첫번째 열인 영화제목 열을 분리
train_name = train_x['영화제목']
test_name = test_x['영화제목']

#직감 ~ 전체 댓글 수 열을 학습에 사용하기 위해 관객 수 열을 분리 
train_x = train_x.iloc[:,1:-1]
test_x = test_x.iloc[:,1:-1]

#텐서플로우 Session 변수 생성 
sess = tf.Session()

#Input Layer 
X = tf.placeholder(tf.float32, shape=([None,7]),name="X")

#Output Layer
Y = tf.placeholder(tf.int32, shape=([None,1]),name="Y")

#One-hot encoding : 250만 이상여부에 대한 one-hot encoding
Y_one_hot = tf.one_hot(Y, 2, name="Y_one_hot")  
Y_one_hot = tf.reshape(Y_one_hot, [-1, 2])

#Hidden1_Layer
W1 = tf.Variable(tf.truncated_normal([7,3]),name='W1')
        
b1 = tf.Variable(tf.truncated_normal([3]),name='b1')


H1_logits = tf.matmul(X, W1) + b1

#Hidden2_Layer
W2 = tf.Variable(tf.truncated_normal([3,2]),name="W2")

b2 = tf.Variable(tf.truncated_normal([2]), name='b2')

logits = tf.matmul(H1_logits, W2) + b2

#softmax 함수를 사용하여 classification
hypothesis = tf.nn.softmax(logits)
            
#오차 정도 계산(cost)
cost_i = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=Y_one_hot)
            
cost = tf.reduce_mean(cost_i)

#학습 및 최적화 : 경사하강법(GradientDesecent), 학습률(0.05)    
optimization = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(cost)

prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            
sess.run(tf.global_variables_initializer())

#설계한 모델 학습 
for step in range(2001):   
    _ = sess.run(optimization, feed_dict={X:train_x,Y:train_y}) 
    
    if step % 1000 == 0:
        loss, acc = sess.run([cost, accuracy], feed_dict={X: train_x, Y: train_y})
        print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(step, loss, acc))

#학습한 결과로 테스트하기 
test_acc,test_predict,test_correct = sess.run([accuracy,prediction,correct_prediction], feed_dict={X: test_x, Y: test_y})
print("Test Prediction =", test_acc)
        
pd.set_option('display.unicode.east_asian_width', True)
sub = pd.DataFrame()

sub['Movie Title'] = test_name
sub['Predict_Audience'] = test_predict
sub['Origin_Audience'] = test_y
sub['Correct'] = test_correct



print(sub)
    

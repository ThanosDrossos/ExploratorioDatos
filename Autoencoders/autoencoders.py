# From:
# https://dropsofai.com/autoencoders-in-keras-and-deep-learning/

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import argparse, sys
	
from tensorflow.keras.datasets import mnist

 
(trainX, trainy), (testX, testy) = mnist.load_data()

print('Training data shapes: X=%s, y=%s' % (trainX.shape, trainy.shape))
print('Testing data shapes: X=%s, y=%s' % (testX.shape, testy.shape))

for j in range(5):
	i = np.random.randint(0, 1000)
	plt.subplot(550 + 1 + j)
	plt.imshow(trainX[i], cmap='gray')
	plt.title(trainy[i])
#plt.show()

# normalizing pixel intensities
trainX = trainX/255
testX = testX/255
#reshaping data into single 784ension
train_data = np.reshape(trainX, (60000, 28*28))
test_data = np.reshape(testX, (10000, 28*28))

print (train_data.shape, test_data.shape)
#cc = sys.stdin.read(1)


import tensorflow

input_data = tensorflow.keras.layers.Input(shape=(784))
 
print ("encoder")
encoder = tensorflow.keras.layers.Dense(100)(input_data)
encoder = tensorflow.keras.layers.Activation('relu')(encoder)
 
encoder = tensorflow.keras.layers.Dense(50)(encoder)
encoder = tensorflow.keras.layers.Activation('relu')(encoder)
 
encoder = tensorflow.keras.layers.Dense(25)(encoder)
encoder = tensorflow.keras.layers.Activation('relu')(encoder)
 
encoded = tensorflow.keras.layers.Dense(2)(encoder)

print ("decoder")
decoder = tensorflow.keras.layers.Dense(25)(encoded)
decoder = tensorflow.keras.layers.Activation('relu')(decoder)
 
decoder = tensorflow.keras.layers.Dense(50)(decoder)
decoder = tensorflow.keras.layers.Activation('relu')(decoder)
 
decoder = tensorflow.keras.layers.Dense(100)(decoder)
decoder = tensorflow.keras.layers.Activation('relu')(decoder)
 
decoded = tensorflow.keras.layers.Dense(784)(decoder)


autoencoder = tensorflow.keras.models.Model(inputs=input_data, outputs=decoded)
autoencoder.compile(loss='mse', optimizer='adam')
autoencoder.summary()

autoencoder.fit(train_data, train_data, epochs=20, batch_size=64, validation_data=(test_data, test_data))

for i in range(5):
    plt.subplot(550 + 1 + i)
    plt.imshow(testX[i], cmap='gray')
#plt.show()

# Reconstructed Images
for i in range(5):
	plt.subplot(550 + 1 + i)
	output = autoencoder.predict(np.array([test_data[i]]))
	op_image = np.reshape(output[0]*255, (28, 28))
	plt.imshow(op_image, cmap='gray')
#plt.show()

#cc = sys.stdin.read(1)


dr_model = tensorflow.keras.models.Model(inputs=autoencoder.get_layer('input_1').input, outputs=autoencoder.get_layer('dense_3').output)
#dr_model = tensorflow.keras.models.Model(inputs=autoencoder.get_layer('input_3').input, outputs=autoencoder.get_layer('dense_14').output)
dr_model.summary()

x = []
y = []
z = []

#Lab = np.loadtxt(args.l)
#for i in range(10000):
for i in range(500):
#for i in range(ln):
    z.append(testy[i])
    op = dr_model.predict(np.array([test_data[i]]))
    x.append(op[0][0])
    y.append(op[0][1])
 
df = pd.DataFrame()
df['x'] = x
df['y'] = y
df['z'] = ["digito-"+str(k) for k in z]

#print ("x = ", df['x'])
#print ("y = ", df['y'])
 
plt.figure(figsize=(8, 6))
plt.xlim([min(df['x']), max(df['x'])])
plt.ylim([min(df['y']), max(df['y'])])
sns.scatterplot(x='x', y='y', hue='z', data=df)
plt.savefig("digits_code.png")
plt.show()

#X = np.loadtxt(args.i)


df.to_csv(args.o + ".csv", sep = '\t')

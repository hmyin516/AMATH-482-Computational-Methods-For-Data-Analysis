#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix


# In[2]:


fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()


# In[3]:


plt.figure()
for k in range(10):
    plt.subplot(3,4,k+1)
    plt.imshow(X_train_full[k], cmap="gray")
    plt.axis('off')
plt.show()


# In[4]:


X_valid = X_train_full[:5000] / 255.0
X_train = X_train_full[5000:] / 255.0
X_test = X_test / 255.0

y_valid = y_train_full[:5000]
y_train = y_train_full[5000:]


# In[5]:


from functools import partial

my_dense_layer = partial(tf.keras.layers.Dense, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.0001))

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=[28, 28]),
    
    my_dense_layer(128),
    #my_dense_layer(64),
    my_dense_layer(32),
    #my_dense_layer(128),
   
    #my_dense_layer(400),
   # my_dense_layer(100),
   # my_dense_layer(100),

    
    #my_dense_layer(600),
    #my_dense_layer(400),
    #my_dense_layer(200),
    my_dense_layer(10, activation="softmax")
])


# In[6]:


model.compile(loss="sparse_categorical_crossentropy",
             optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
             metrics=["accuracy"])


# validation below training -> not overfitting. validation go down whole time, train further. 

# In[7]:


history = model.fit(X_train, y_train, epochs=40, validation_data=(X_valid,y_valid))


# In[8]:


pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.ylabel("Percentage")
plt.xlabel("Epoch")
plt.show()


# In[9]:


model.evaluate(X_test,y_test)


# In[10]:


y_pred = model.predict_classes(X_train)
conf_train = confusion_matrix(y_train, y_pred)
print(conf_train)


# In[11]:


y_pred = model.predict_classes(X_test)
conf_test = confusion_matrix(y_test, y_pred)
print(conf_test)


# CNN

# In[12]:


fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
X_valid = X_train_full[:5000] / 255.0
X_train = X_train_full[5000:] / 255.0
X_test = X_test / 255.0

y_valid = y_train_full[:5000]
y_train = y_train_full[5000:]


# In[29]:


plt.figure()
for k in range(10):
    plt.subplot(3,4,k+1)
    plt.imshow(X_train_full[k], cmap="gray")
    plt.axis('off')
    plt.title(y_train_full[k])
plt.show()


# In[13]:


X_train = X_train[..., np.newaxis]
X_valid = X_valid[..., np.newaxis]
X_test = X_test[..., np.newaxis]


# In[14]:


my_dense_layer = partial(tf.keras.layers.Dense, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.0001))
my_conv_layer = partial(tf.keras.layers.Conv2D, activation="relu", padding="valid")

model = tf.keras.models.Sequential([
    my_conv_layer(6,5,padding="same",input_shape=[28,28,1]),
    tf.keras.layers.AveragePooling2D(2),
   # my_conv_layer(8,5),
    #tf.keras.layers.AveragePooling2D(2),
    my_conv_layer(32,3),
    my_conv_layer(64,3),
    my_conv_layer(128,3),
    my_conv_layer(256,3),
    my_conv_layer(512,3),
    tf.keras.layers.Flatten(),
    my_dense_layer(256),
    #my_dense_layer(32),
    #my_dense_layer(8),
    my_dense_layer(10, activation="softmax")
])


# In[15]:


model.compile(loss="sparse_categorical_crossentropy",
             optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
             metrics=["accuracy"])


# In[16]:


history = model.fit(X_train, y_train, epochs=10, validation_data=(X_valid,y_valid))


# In[17]:


pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.ylabel("Percentage")
plt.xlabel("Epoch")
plt.show()


# In[18]:


y_pred = model.predict_classes(X_train)
conf_train = confusion_matrix(y_train, y_pred)
print(conf_train)


# In[19]:


model.evaluate(X_test,y_test)


# In[20]:


y_pred = model.predict_classes(X_test)
conf_test = confusion_matrix(y_test, y_pred)
print(conf_test)


# In[ ]:





# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import numpy as np


# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
# classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
# classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
# classifier.add(MaxPooling2D(pool_size=(2, 2)))
# classifier.add(Dropout(0.25))
# classifier.add(Conv2D(64, (3, 3), activation='relu'))
# classifier.add(Conv2D(64, (3, 3), activation='relu'))
# classifier.add(MaxPooling2D(pool_size=(2, 2)))
# classifier.add(Dropout(0.25))
# classifier.add(Flatten())
# classifier.add(Dense(500, activation='relu'))
# classifier.add(Dropout(0.5))
# classifier.add(Dense(units = 3, activation = 'softmax'))

classifier.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(64, 64, 3)))
classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.25))

classifier.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
classifier.add(Conv2D(64, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.25))

classifier.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
classifier.add(Conv2D(64, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.25))

classifier.add(Flatten())
classifier.add(Dense(512, activation='relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(3, activation='softmax'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


# Part 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

# train_datagen = ImageDataGenerator(rescale = 1./255)

# 데이터셋 불러오기
train_datagen = ImageDataGenerator(rescale=1./255, 
                                   rotation_range=10,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.7,
                                   zoom_range=[0.9, 2.2],
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   fill_mode='nearest')

# test_datagen = ImageDataGenerator(rescale = 1./255)

test_datagen = ImageDataGenerator(rescale=1./255, 
                                   rotation_range=10,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.7,
                                   zoom_range=[0.9, 2.2],
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   fill_mode='nearest')


training_set = train_datagen.flow_from_directory('data/train',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

valid_set = test_datagen.flow_from_directory('data/validation',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'categorical')

#test_set = test_datagen.flow_from_directory('data/validation',
#                                           target_size = (64, 64),
#                                           batch_size = 12,
#                                           class_mode = 'categorical')

hist = classifier.fit_generator(training_set,
                         steps_per_epoch = 100,
                         epochs = 20,
                         validation_data = valid_set,
                         validation_steps = 200)


# output = classifier.predict_generator(test_set, steps=5)
# print(test_set.class_indices)
# print(output)

# 모델 평가하기
print("-- Evaluate --")

scores = classifier.evaluate_generator(
            valid_set, 
            steps = 10)

print("%s: %.2f%%" %(classifier.metrics_names[1], scores[1]*100))

# 모델 예측하기
print("-- Predict --")

output = classifier.predict_generator(
            valid_set, 
            steps = 10)
print(valid_set.class_indices)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

print(output)
print(valid_set.filenames)



# 5. 학습과정 살펴보기
import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
#loss_ax.set_ylim([0.0, 0.5])

acc_ax.plot(hist.history['acc'], 'b', label='train acc')
acc_ax.plot(hist.history['val_acc'], 'g', label='val acc')
#acc_ax.set_ylim([0.8, 1.0])

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuray')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()

import coremltools
# Saving the Core ML model to a file.
#coreml_model = coremltools.converters.keras.convert(classifier)
#class_labels = ['cherry', 'hanlabong', 'lemon', 'raspberry', 'strawberry', 'tangerine']
class_labels = ['lemon', 'strawberry', 'tangerine']
coreml_model = coremltools.converters.keras.convert(classifier, input_names='image', image_input_names='image', class_labels=class_labels, is_bgr=True)  
coreml_model.save('my_model.mlmodel')


from keras.models import load_model

classifier.save('fruit_cnn_keras_model.h5')


from keras.models import load_model
from keras.preprocessing import image

classifier = load_model('fruit_cnn_keras_model.h5')
classifier.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# predicting multiple images at once
img = image.load_img('lemon/000001.jpg', target_size=(64, 64))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
y = image.img_to_array(img)
y = np.expand_dims(y, axis=0)

# pass the list of multiple images np.vstack()
images = np.vstack([x, y])
#classes = classifier.predict_classes(images, batch_size=32.)
classes = classifier.predict_classes(images, batch_size=1, verbose=1)

# print the classes, the images belong to
print(classes)
#print(classes[0])
#print(classes[1])

img = image.load_img('lemon/000002.jpg', target_size=(64, 64))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
y = image.img_to_array(img)
y = np.expand_dims(y, axis=0)

# pass the list of multiple images np.vstack()
images = np.vstack([x, y])
classes = classifier.predict_classes(images, batch_size=32, verbose=1)

# print the classes, the images belong to
print(classes)
#print(classes[0][0])
#print(classes[1])


img = image.load_img('lemon/000003.jpg', target_size=(64, 64))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
y = image.img_to_array(img)
y = np.expand_dims(y, axis=0)

# pass the list of multiple images np.vstack()
images = np.vstack([x, y])
classes = classifier.predict_classes(images, batch_size=32, verbose=1)

# print the classes, the images belong to
print(classes)
#print(classes[0])
#print(classes[1])


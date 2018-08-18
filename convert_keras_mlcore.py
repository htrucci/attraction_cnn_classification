from keras.models import load_model
model = load_model('./cnn_attraction_keras_model.h5')

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=10,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.7,
                                   zoom_range=[0.9, 2.2],
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   fill_mode='nearest',
                                   validation_split=0.33)

training_set = train_datagen.flow_from_directory('/floyd/input/data',
                                                 shuffle=True,
                                                 seed=13,
                                                 target_size = (224, 224),
                                                 batch_size = 15,
                                                 class_mode = 'categorical',
                                                 subset="training")
label_map = (training_set.class_indices)
print(label_map)

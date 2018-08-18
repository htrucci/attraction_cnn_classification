from keras.models import load_model
import coremltools
model = load_model('./cnn_attraction_keras_model.h5')
# Saving the Core ML model to a file.
class_labels = ['colosseum', 'eiffel', 'liberty', 'niagara', 'pyramid']
coreml_model = coremltools.converters.keras.convert(model, input_names='image', image_input_names='image', class_labels=class_labels, is_bgr=True)
coreml_model.save('./cnn_attraction_model.mlmodel')

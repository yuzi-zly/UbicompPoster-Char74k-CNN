from tensorflow.keras.models import load_model
from tensorflow.python.keras import backend as k
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

model = load_model('./model/char74k_cnn.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open("./model/char74k_cnn.tflite", "wb").write(tflite_model)

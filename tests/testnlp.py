
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import tensorflow as tf
import tensorflow_hub as hub

def test_models(model_name):
    # returns a compiled model
    # identical to the previous one
    model = tf.keras.models.load_model(model_name,custom_objects={'KerasLayer':hub.KerasLayer})

    print("="*100)
    print("Start Testing...............")
    print("="*100)
    df = ["The movie was great!",
          "The movie was okay.",
          "The movie was terrible...",
         "The movie was great nice good..."]

    ts = tf.convert_to_tensor(df, dtype=tf.string)
    pred = model.predict(ts)
    for i in range(len(pred)):
        conf = pred[i][0]
        predict = "positive" if conf>0.5 else "negative"
        print("conf: ", conf, ", sentimet: ", predict)





















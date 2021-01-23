
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def train_evaluate_pretrained_models(embedding, save_model_name, shuffle, epochs, batch_size, train_ds, val_ds, test_ds):
    import tensorflow as tf
    import tensorflow_hub as hub
    
    hub_layer = hub.KerasLayer(embedding, input_shape=[],
                           dtype=tf.string, trainable=False)
    model = tf.keras.Sequential()
    model.add(hub_layer)
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    # model.add(tf.keras.layers.Dense(512, activation='relu'))
    # model.add(tf.keras.layers.Dense(128, activation='relu'))
    # model.add(tf.keras.layers.Dropout(0.1))
    #model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
    model.summary()
    # Compiling
    model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
    
    history = model.fit(train_ds.shuffle(shuffle).batch(batch_size),
                    epochs=epochs,
                    validation_data=val_ds.batch(batch_size),
                    verbose=1)

    # Evaluation
    results1 = model.evaluate(val_ds.batch(batch_size), verbose=2)
    for name1, value1 in zip(model.metrics_names, results1):
      print("%s: %.3f" % (name1, value1))
    

    
    history_dict = history.history

    results2 = model.evaluate(test_ds.batch(batch_size), verbose=2)

   
    model.save(save_model_name)
    del model  # deletes the existing model
    
    return {
        "acc": history_dict['accuracy'],
        "val_acc": history_dict['val_accuracy'],
        "loss" : history_dict['loss'],
        "val_loss":  history_dict['val_loss'],
        "Test Loss": results2[0],
        "Test Accuracy": results2[1]
    }


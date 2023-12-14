# -*- coding: utf-8 -*-

import tensorflow as tf
import os
import datetime
import numpy as np
from helpers import save_training_history_to_file, plot_model_and_save, make_directory, calculate_input_shape, plot_heat_map, load_data, plot_history_tf, count_classes, list_class_names, load_ex_data

project_path = "./"
model_name = "layer1"
model_filename = model_name + ".h5"
model_dir = model_name + "_model"
logs_dir = project_path + "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
model_path = project_path + model_dir + "/" + model_filename

RANDOM_SEED = 42
NUM_EPOCHS = 30
BATCH_SIZE = 16
RATIO = 0.4

def build_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Reshape(target_shape=(input_shape[0], 1), input_shape=input_shape),
        tf.keras.layers.Conv1D(filters=16, kernel_size=7, strides=2, padding='SAME', activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=2, strides=1, padding='SAME'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.01),
        tf.keras.layers.Dense(count_classes(), activation='softmax')
    ])
    
    return model



def main():
    # X_train,y_train is the training set
    # X_test,y_test is the test set
    X_train, X_test, y_train, y_test = load_data(RATIO, RANDOM_SEED)
    ex_X_train, ex_X_test, ex_y_train, ex_y_test = load_ex_data(RATIO, RANDOM_SEED)
    input_shape = calculate_input_shape(X_train)

    if os.path.exists(model_path):
        # import the pre-trained model if it exists
        print('-------------------------------------------------------')
        print('Import the pre-trained model, skip the training process')
        print('-------------------------------------------------------')
        model = tf.keras.models.load_model(filepath=model_path)
        

    
    else:
        # build the CNN model
        model = build_model(input_shape)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0004768118483483392)
        model.compile(optimizer=optimizer,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        model.summary()
        
        

        
        # Train and evaluate model
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logs_dir, histogram_freq=1)
        history = model.fit(X_train, y_train, epochs=NUM_EPOCHS,
                            batch_size=BATCH_SIZE,
                            validation_data=(X_test, y_test),
                            callbacks=[tensorboard_callback])
        # Save training history to a text file
        save_training_history_to_file(history, model_name,  NUM_EPOCHS=NUM_EPOCHS)

        # Save the model
        model.save(filepath=model_path)

        # Plot the training history
        plot_history_tf(history, model_name)

        # Predict the class of test data
        y_pred = np.argmax(model.predict(X_test), axis=-1)
        # Plot confusion matrix heat map
        plot_heat_map(y_test, y_pred, model_name)
    
        # Plot and save the model architecture
        plot_model_and_save(model, model_name)

if __name__ == '__main__':
    main()

import pathlib
import os
import pandas as pd
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
import numpy as np
import seaborn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import shutil
import datetime
import tensorflow as tf
from sklearn.manifold import TSNE
from tensorflow.keras.utils import plot_model

verbose = False

tag = 'default'

# パスの設定
model_directory_path = 'models'
model_suffix = '.joblib'
data_directory_path = 'data'
data_suffix = '_porod.csv'
ex_data_suffix = '_Iq.csv'
data_prefix = 'random_'

# 現在のモデルディレクトリへのパス
def get_model_directory_path():
    return pathlib.Path(model_directory_path, tag)

def get_model_file_path(filename):
    return pathlib.Path(get_model_directory_path(), filename)

def log_verbose(*args):
    if verbose:
        print(*args)

# ファイルの入出力

def preprocess_input_file(filename):
    input_df = pd.read_csv(filename)
    # input_df.rename(columns={input_df.columns[0]: 'id'}, inplace=True)
    return input_df

def get_data_files():
    return pathlib.Path(data_directory_path).glob('*' + data_suffix)

def get_ex_data_files():
    return pathlib.Path(data_directory_path).glob('*' + ex_data_suffix)


def load_data(ratio, random_seed):
    print('------------load_data------------')
    content = []
    for i, txt_file in enumerate(get_data_files()):
        filename = os.path.basename(txt_file)
        print('filename:',filename)
        # read in data
        shape = preprocess_input_file(txt_file)
        #add 'shape' column with shape name
        shape['label'] = i
        content.append(shape)
        print('label:',i)
    all_df = pd.concat(content, axis=0, ignore_index=True)
    
    df_new = all_df.iloc[:,1:]
    
    x_data=df_new.iloc[:,:-1].values
    
    y_data=df_new["label"].values
    
    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data,test_size=ratio,random_state=random_seed)
    
    return X_train, X_test, y_train, y_test

def load_ex_data(ratio, random_seed):
    print('------------load_ex_data------------')
    content = []
    for i, txt_file in enumerate(get_ex_data_files()):
        filename = os.path.basename(txt_file)
        print('filename:',filename)
        # read in data
        shape = preprocess_input_file(txt_file)
        #add 'shape' column with shape name
        shape['label'] = i
        content.append(shape)
        print('label:',i)
    all_df = pd.concat(content, axis=0, ignore_index=True)
    
    df_new = all_df.iloc[:,1:]
    
    x_data=df_new.iloc[:,:-1].values
    
    y_data=df_new["label"].values
    
    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data,test_size=ratio,random_state=random_seed)
    
    return X_train, X_test, y_train, y_test

def calculate_input_shape(X_train):
    input_shape = X_train.shape[1:]
    return input_shape


def list_class_names():
    classes =[]
    for txt_file in get_data_files():
        filename = os.path.basename(txt_file)
        classes.append(filename[len(data_prefix):-len(data_suffix)])
    return classes

def count_classes():
    count = sum(1 for _ in get_data_files())
    return count

# ユーティリティ関数

def make_directory(model_name, subdir_name):
    base_dir = "./output"  # Specify the base directory to save the output
    model_dir = os.path.join(base_dir, model_name)
    subdir_path = os.path.join(model_dir, subdir_name)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if not os.path.exists(subdir_path):
        os.makedirs(subdir_path)

    return subdir_path

# 可視化


def plot_heat_map(y_test, y_pred, model_name):


    con_mat = confusion_matrix(y_test, y_pred)
    classes = []
    num = []
    for j, txt_file in enumerate(get_data_files()):
        filename = os.path.basename(txt_file)
        classes.append(filename[len(data_prefix):-len(data_suffix)])
        num.append(j + 0.5)

    plt.figure(figsize=(8, 8))
    seaborn.heatmap(con_mat, annot=True, fmt='.20g', cmap='Blues')
    plt.ylim(0, len(classes))
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title(f'{model_name} Confusion Matrix')
    plt.xticks(num, classes, rotation=90)
    plt.yticks(num, classes, rotation=0)

    save_dir = make_directory(model_name, 'confusion_matrix')
    save_path = os.path.join(save_dir, 'confusion_matrix.jpeg')
    plt.savefig(save_path)
    plt.show()




def plot_history_tf(history, model_name):
    

    plt.figure(figsize=(8, 8))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'{model_name} Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    
    save_dir = make_directory(model_name, 'accuracy')
    save_path = os.path.join(save_dir, 'accuracy.jpeg')
    plt.savefig(save_path)
    plt.show()

    plt.figure(figsize=(8, 8))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'{model_name} Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    
    save_dir = make_directory(model_name, 'loss')
    save_path = os.path.join(save_dir, 'loss.jpeg')
    plt.savefig(save_path)
    plt.show()
    
def save_training_history_to_file(history, model_name,  NUM_EPOCHS):
    # Create output directories
    subdir_path = make_directory(model_name, "training_history")
    output_file_path = os.path.join(subdir_path, "training_history.txt")
    with open(output_file_path, "w") as f:
        f.write(f"model name : {model_name}\n")
        for epoch in range(NUM_EPOCHS):
            f.write(f"Epoch {epoch + 1}/{NUM_EPOCHS}\n")
            f.write(f"Train - loss: {history.history['loss'][epoch]}, accuracy: {history.history['accuracy'][epoch]}\n")
            f.write(f"Validation - loss: {history.history['val_loss'][epoch]}, accuracy: {history.history['val_accuracy'][epoch]}\n")
            f.write("\n")

def plot_model_and_save(model, model_name):
    

    save_dir = make_directory(model_name, 'architecture')
    save_path = os.path.join(save_dir, 'architecture.jpeg')

    plot_model(model, to_file=save_path, show_shapes=True, show_layer_names=True, rankdir='TB')

    print(f"Model architecture saved at: {save_path}")



    

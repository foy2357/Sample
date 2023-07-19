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



    

def visualize_selected_indices(model, model_name, X_test, y_test, ex_X_test, layer_name, selected_label, num_threshold=15, pred_threshold=0.9):


    def grad_cam(layer_name, data):
        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(layer_name).output, model.output]
        )
        last_conv_layer_output, preds = grad_model(data)

        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(data)
            pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0))
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output * pooled_grads
        heatmap = tf.reduce_mean(heatmap, axis=(1))
        heatmap = (heatmap - tf.reduce_min(heatmap)) / (tf.reduce_max(heatmap) - tf.reduce_min(heatmap))
        heatmap = np.expand_dims(heatmap, 0)
        return heatmap

    class_labels = list_class_names()
    print("Class labels:", class_labels)
    
    def get_label_index(label):
        if label in class_labels:
            return class_labels.index(label)
        else:
            return None
    
    selected_index = get_label_index(selected_label)
    
    cnt = 0
    num_selected = 0
    selected_indices = []  # pred > pred_threshold を満たすインデックスを保存するリスト

    base_dir = "./output"  # Specify the base directory to save the output
    model_dir = os.path.join(base_dir, model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    for index, data in enumerate(X_test):
        data = np.expand_dims(data, 0)
        pred = model.predict(data)
        pred = pred[0][selected_index]

        if pred > pred_threshold:
            selected_indices.append(index)  # pred > pred_threshold を満たすインデックスを保存

            heatmap = grad_cam(layer_name, data)
            selected_name = class_labels[selected_index]
            true_label = class_labels[int(y_test[cnt])]

            print(f"Model prediction: {selected_name} ({pred}), True label: {true_label}")

            extra_data = ex_X_test[index]  # 対応するインデックスのデータを取得
            extra_data = np.expand_dims(extra_data, 0)

            input_shape = X_test.shape[1]  # X_testのinput_shapeを取得

            subdir_path = make_directory(model_name, f"{selected_name}_{index}")
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12))

            im1 = ax1.imshow(
                np.expand_dims(heatmap, axis=2),
                cmap="jet",
                aspect="auto",
                interpolation="nearest",
                extent=[0, input_shape, extra_data.min() - 1, extra_data.max() + 1],
                alpha=0.5,
            )

            ex_data = ex_X_test[index]
            ax1.plot(ex_data, "k")  # データをプロット
            ax1.set_xlabel(r"$\log q$")
            ax1.set_ylabel(r"$\log I$")
            ax1.set_yscale('log')  # Set y-axis scale to logarithmic

            data = X_test[index]

            im2 = ax2.imshow(
                np.expand_dims(heatmap, axis=2),
                cmap="jet",
                aspect="auto",
                interpolation="nearest",
                extent=[0, input_shape, data.min() - 1, data.max() + 1],
                alpha=0.5,
            )
            data = X_test[index]
            ax2.plot(data, "k")
            ax2.set_xlabel(r"$\log q$")
            ax2.set_ylabel(r"$\frac{d\log I}{d\log q}$")

            fig.colorbar(im1, ax=ax1)
            fig.colorbar(im2, ax=ax2)
            plt.subplots_adjust(hspace=0.4)
            # plt.title(f"Model prediction: {selected_name} ({pred}), True label: {true_label}", pad=20)
            plt.show()

            # Save the figure using the save path
            save_path = os.path.join(subdir_path, f"{selected_name}_{index}.jpeg")
            fig.savefig(save_path, dpi=300)
            plt.close(fig)

            # Save the model prediction information in a text file
            text_content = f"Model prediction: {selected_name} ({pred}), True label: {true_label}"
            text_file_path = os.path.join(subdir_path, f"{selected_name}_{index}.txt")
            with open(text_file_path, 'w') as text_file:
                text_file.write(text_content)

            num_selected += 1
        cnt += 1
        if num_selected > num_threshold:
            break

    print("Selected indices:", selected_indices)  # pred > pred_threshold を満たすインデックスの出力



def visualize_features(model, model_name,  X_test, y_test, layer_names):
    intermediate_layers = [model.get_layer(layer_name).output for layer_name in layer_names]
    intermediate_models = tf.keras.models.Model(inputs=model.input, outputs=intermediate_layers)

    intermediate_features = intermediate_models.predict(X_test)

    plt.figure(figsize=(12, 4))

    tsne = TSNE(n_components=2, random_state=42)
    for i, features in enumerate(intermediate_features):
        tsne_features = tsne.fit_transform(features.reshape((X_test.shape[0], -1)))
        ax = plt.subplot(1, len(intermediate_features), i + 1)
        scatter = ax.scatter(tsne_features[:, 0], tsne_features[:, 1], c=y_test, cmap='viridis')
        plt.title(f'{layer_names[i]} Features')

        # Add class labels to the scatter plot
        handles, labels = scatter.legend_elements()
        legend = ax.legend(handles, list_class_names(), loc='lower right', title='Class')

    plt.tight_layout()


    save_dir = make_directory(model_name, 't-SNE')
    save_path = os.path.join(save_dir, 't-SNE.jpeg')
    plt.savefig(save_path)

    print(f"Figure saved at: {save_path}")



    
def single_visualize_feature(model, model_name,  X_test, y_test, single_layer_name):
    intermediate_layer = model.get_layer(single_layer_name).output
    intermediate_model = tf.keras.models.Model(inputs=model.input, outputs=intermediate_layer)

    plt.figure(figsize=(6, 6))

    tsne = TSNE(n_components=2, random_state=42)
    tsne_features = tsne.fit_transform(intermediate_model.predict(X_test).reshape((X_test.shape[0], -1)))

    scatter = plt.scatter(tsne_features[:, 0], tsne_features[:, 1], c=y_test, cmap='viridis')
    plt.title(f'{single_layer_name} Features')

    handles, labels = scatter.legend_elements()
    legend = plt.legend(handles, list_class_names(), loc='upper right', title='Class', bbox_to_anchor=(1.2, 1))

    plt.tight_layout()


    save_dir = make_directory(model_name, 'single_t-SNE')
    save_path = os.path.join(save_dir, 'single_t-SNE.jpeg')
    plt.savefig(save_path)

    print(f"Figure saved at: {save_path}")





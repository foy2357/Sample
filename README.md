# CNN Model Training and Helpers

このリポジトリは、畳み込みニューラルネットワーク (CNN) モデルを訓練し、スペクトル分類タスクを行うための Python スクリプトとサポートモジュールを含んでいます。`cnn_model_training.py` スクリプトは、データのロードからモデルの構築、訓練、評価までを自動化し、`helpers.py` モジュールは訓練プロセスをサポートする便利な関数を提供します。

## 必要な環境

以下の環境が必要です：

- Python 3.11.4
- TensorFlow (バージョン 2.7.0)
- pandas (バージョン 1.4.1)
- scikit-learn (バージョン 1.0.2)
- numpy (バージョン 1.22.2)
- seaborn (バージョン 0.11.2)
- matplotlib (バージョン 3.5.1)

これらのライブラリは、`cnn_model_training.py` および `helpers.py` ファイルの先頭に `import` されているものです。実行前にこれらのライブラリをインストールしてください。

## ファイル構造

このリポジトリのファイル構造は以下のようになっています：

```
- data/
  - (ここにデータセットを配置)
- model/
  - default.h5
- output/
  - default/
    - training_history/
      - training_history.txt
    - architecture/
      - architecture.jpeg
    - accuracy/
      - accuracy.jpeg
    - loss/
      - loss.jpeg
    - confusion_matrix/
      - confusion_matrix.jpeg
    - single_t-SNE/
      - single_t-SNE.jpeg
    - t-SNE/
      - t-SNE.jpeg
- cnn_model_training.py
- helpers.py
- LICENSE.md
- README.md
```

## 使用方法

1. データセットを適切なディレクトリ構造で `data` フォルダに配置します。ファイルの命名規則に注意してください。

2. `cnn_model_training.py` のハイパーパラメータを必要に応じて調整します。例えば、エポック数、バッチサイズ、学習率などを変更します。

3. 以下のコマンドを実行して、CNN モデルを訓練します。

```
python cnn_model_training.py
```

4. 訓練が完了すると、以下のファイルとディレクトリが生成されます：

- 訓練されたモデル: `./model/default.h5`
- モデルの訓練履歴テキストファイル: `./output/default/training_history/training_history.txt`
- モデルの可視化アーキテクチャ: `./output/default/architecture/architecture.jpeg`
- モデルの訓練履歴の可視化（損失と精度）: `./output/default/accuracy/accuracy.jpeg` および `./output/default/loss/loss.jpeg`
- モデルの混同行列ヒートマップ: `./output/default/confusion_matrix/confusion_matrix.jpeg`
- 畳み込み層の特徴マップのt-SNE可視化: `./output/default/single_t-SNE/single_t-SNE.jpeg`　および `./output/default/t-SNE/t-SNE.jpeg`

## `helpers.py`

`helpers.py` モジュールは、`cnn_model_training.py` で使用されるサポート関数を提供します。主な機能は以下の通りです：

- データの前処理とロード
- ファイルの入出力とディレクトリ作成
- モデルのアーキテクチャと特徴量の可視化

`cnn_model_training.py` の訓練プロセスを補助する機能を提供しています。

## `helpers.py` /`visualize_selected_indices` 関数

`visualize_selected_indices` 関数は、スペクトル分類モデルが正確に予測したスペクトルデータの中から、特定のクラスに属するデータを視覚化します。関数の主な入力は以下です：

- `model`: 訓練されたCNNモデル
- `model_name`: モデルの名前
- `X_test`: テストデータのスペクトルデータ
- `y_test`: テストデータの真のクラスラベル
- `single_layer_name`: CNNモデルの特定のレイヤー名
- `selected_label`: 視覚化したいクラスのラベル
- `num_threshold`: 視覚化するデータポイントの上限数
- `pred_threshold`: データポイントを選択するための予測確率の閾値

`visualize_selected_indices` 関数は、予測確率が `pred_threshold` 以上であるデータポイントを特定し、それらのスペクトルデータと特定のレイヤーの特徴マップを視覚化します。これにより、モデルが特定のクラスを予測する際の重要なスペクトルパターンを可視化することができます。

## ライセンス

このプロジェクトはMITライセンスの下で利用できます。詳細は [LICENSE.md](LICENSE.md) を参照してください。

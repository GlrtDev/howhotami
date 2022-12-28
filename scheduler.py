from bgFeatureExtraction import bgExtractor
import cv2
from os import listdir
from os.path import isfile, join
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error , r2_score, max_error
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import svm, linear_model, preprocessing , utils
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import joblib
from keras import backend as K


def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [MPG]')
    plt.legend()
    plt.grid(True)

def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true - y_pred )) 
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

class Scheduler:

    def __init__(self) -> None:
        self.db_path = 'howhotami/SCUT-FBP5500_v2'

    def load_img_paths(self):
        files = [f for f in listdir(f'{self.db_path}/Images') if isfile(join(f'{self.db_path}/Images', f))]
        return files

    def load_img_labels(self):
        excel_df = pd.read_excel(f'{self.db_path}/All_Ratings.xlsx', sheet_name='ALL', usecols=['Filename', 'Rating'])
        return excel_df



    def run(self, transfer_learning = False):
        
        if not transfer_learning:
            print('starting color regression...')
            if isfile('output.csv'):
                print('found data file!')
                dataset = np.genfromtxt('output.csv', delimiter=',', dtype =np.float32)
            else:
                ratings_df = self.load_img_labels()
                ratings_df_grouped = ratings_df.groupby('Filename').mean()
                imgs_filenames = self.load_img_paths()
                main_colors = dict()
                with ThreadPoolExecutor(max_workers=12) as executor:
                    futures = []
                    for index, img_filename in enumerate(imgs_filenames):
                        img = cv2.imread(f'{self.db_path}/Images/{img_filename}')
                        if index%100 == 0:
                            print(f'{(100*index/5500):.2f}% loaded')
                        futures.append(executor.submit(bgExtractor.run, img, img_filename))
                        
                    for index, future in enumerate(futures):
                        color, filename = future.result()
                        main_colors[filename] = color
                        if index%20 == 0:
                            print(f'{(100*index/5500):.2f}% proccessed')

                w = csv.writer(open("output.csv", "w"))
                dataset = []
                for key, val in main_colors.items():
                    val = [col_v for col in val for col_v in col] #flatten array
                    val.append(ratings_df_grouped.at[key, 'Rating'])
                    dataset.append(val)
                    w.writerow(val)
                dataset = np.array(dataset, np.float32)

            X = dataset[:,:-1]
            scaler = StandardScaler()
            scaler.fit(X)
            X = scaler.transform(X)

            Y = dataset[:,-1]
            kfold = KFold(n_splits=5, shuffle=True, random_state=43)
            index = 0
            scores = []
            for train, test in kfold.split(X, Y):

                X_train, X_test, y_train, y_test = X[train], X[test], Y[train], Y[test]

                #model = svm.LinearSVR(epsilon = 0.1, C = 0.15, max_iter=2000)
                model = linear_model.SGDRegressor(max_iter=10000)
                
                model.fit(X_train, y_train)
                test_predictions = model.predict(X_test).flatten()

                a = plt.axes(aspect='equal')
                plt.scatter(y_test, test_predictions)
                plt.xlabel('True Values')
                plt.ylabel('Predictions')
                lims = [1, 5]
                plt.xlim(lims)
                plt.ylim(lims)
                plt.plot(lims, lims)
                plt.savefig(f'predict_col_{index}.png')

                scoresR2 = r2_score(y_test, test_predictions)
                scoresMAE = max_error(y_test, test_predictions)
                scoresRMSE = np.sqrt(mean_squared_error(y_test, test_predictions))

                cross_val_r2_scores.append(scoresR2)
                cross_val_MAE_scores.append(scoresMAE)
                cross_val_RMSE_scores.append(scoresRMSE)

                #joblib.dump(model, 'model_color.pkl')
        else:
            print('starting transfer learning...')
            ratings_df = self.load_img_labels()
            ratings_df_grouped = ratings_df.groupby('Filename').mean()
            imgs_filenames = self.load_img_paths()
            dataset = dict()
            for index, img_filename in enumerate(imgs_filenames):
                dataset[img_filename] = cv2.imread(f'{self.db_path}/Images/{img_filename}')
                if index%100 == 0:
                            print(f'{(100*index/5500):.2f}% loaded')
            
            X = list()
            Y = list()
            image_size = (160,160)
            for filename, img in dataset.items():
                X.append(cv2.resize(img, image_size, interpolation = cv2.INTER_AREA))
                Y.append(ratings_df_grouped.at[filename, 'Rating'])
            
            X_raw = np.array(X)
            Y = np.array(Y)

            for model_index in ['mobilenet2', 'resnetv2', 'vgg19', 'xception']:
                cross_val_r2_scores = []
                cross_val_MAE_scores = []
                cross_val_RMSE_scores = []
                
                if model_index == 'resnetv2':
                    X = tf.keras.applications.resnet_v2.preprocess_input(X_raw)
                    base_model = tf.keras.applications.ResNet50V2(include_top=False, input_shape=(image_size[0],image_size[1],3))
                if model_index == 'mobilenet2':
                    X = tf.keras.applications.mobilenet_v2.preprocess_input(X_raw)
                    base_model = tf.keras.applications.MobileNetV2(include_top=False, input_shape=(image_size[0],image_size[1],3))
                if model_index == 'vgg19':
                    X = tf.keras.applications.vgg19.preprocess_input(X_raw)
                    base_model = tf.keras.applications.VGG19(include_top=False, input_shape=(image_size[0],image_size[1],3))
                if model_index == 'xception':
                    X = tf.keras.applications.xception.preprocess_input(X_raw)
                    base_model = tf.keras.applications.Xception(include_top=False, input_shape=(image_size[0],image_size[1],3))
                
                base_model.trainable = False

                kfold = KFold(n_splits=5, shuffle=True, random_state=43)
                index = 0
                for train, test in kfold.split(X, Y):
                    
                    X_test, X_valid, y_test, y_valid = train_test_split(X[test], Y[test], test_size=0.5, random_state=92)
                    X_train = X[train]
                    y_train = Y[train]

                    model2 = keras.Sequential(
                    [
                        layers.BatchNormalization(input_shape = (image_size[0],image_size[1],3)),
                        base_model,
                        layers.BatchNormalization(),
                        layers.GlobalAveragePooling2D(),
                        layers.Dropout(0.2),
                        layers.Dense(1, activation = 'linear')
                    ]
                    )

                    model2.summary()
                    model2.compile(loss='mean_absolute_error',
                        optimizer=tf.keras.optimizers.Adam(0.001),
                        metrics=[r2_keras])
                    history = model2.fit(
                        X_train,
                        y_train,
                        validation_data = (X_valid, y_valid),
                        verbose=2, epochs=100
                        )


                    test_predictions = model2.predict(X_test).flatten()
                    a = plt.axes(aspect='equal')
                    plt.scatter(y_test, test_predictions)
                    plt.xlabel('True Values')
                    plt.ylabel('Predictions')
                    lims = [1, 5]
                    plt.xlim(lims)
                    plt.ylim(lims)
                    plt.plot(lims, lims)
                    plt.savefig(f'predict_cv{index}_{model_index}.png')
                    plt.clf()
                    index += 1

                    acc = history.history['r2_keras']
                    val_acc = history.history['val_r2_keras']
                    loss = history.history['loss']
                    val_loss = history.history['val_loss']
                    plt.figure(figsize=(8, 8))
                    plt.subplot(2, 1, 1)
                    plt.plot(acc, label='Training r2 score')
                    plt.plot(val_acc, label='Validation r2 score')
                    plt.legend(loc='lower right')
                    plt.ylabel('r2 score')
                    plt.ylim([0,1.0])
                    plt.title('Training and Validation r2 score')
                    plt.subplot(2, 1, 2)
                    plt.plot(loss, label='Training Loss')
                    plt.plot(val_loss, label='Validation Loss')
                    plt.legend(loc='upper right')
                    plt.ylabel('loss [MAE]')
                    plt.ylim([0,1.0])
                    plt.title('Training and Validation Loss [MAE]')
                    plt.xlabel('epoch')
                    plt.savefig(f'training_cv{index}_{model_index}.png')
                    plt.clf()
                    
                    scoresR2 = r2_score(y_test, test_predictions)
                    scoresMAE = max_error(y_test, test_predictions)
                    scoresRMSE = np.sqrt(mean_squared_error(y_test, test_predictions))
                    print(f'r2: {scoresR2}')
                    print(f'MAE: {scoresMAE}')
                    print(f'RMSE: {scoresRMSE}')
                    cross_val_r2_scores.append(scoresR2)
                    cross_val_MAE_scores.append(scoresMAE)
                    cross_val_RMSE_scores.append(scoresRMSE)
                print('final')
                print(f'r2: {cross_val_r2_scores}')
                print(f'MAE: {cross_val_MAE_scores}')
                print(f'RMSE: {cross_val_RMSE_scores}')
                f = open(f"{model_index}.txt", "w")
                f.write(f"r2: {cross_val_r2_scores}\nMAE: {cross_val_MAE_scores}\nRMSE: {cross_val_RMSE_scores}")
                f.close()
                model2.save(f"model_{model_index}.h5")


scheduler = Scheduler()
scheduler.run(transfer_learning=True)
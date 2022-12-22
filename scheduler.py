from bgFeatureExtraction import bgExtractor
import cv2
from os import listdir
from os.path import isfile, join
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import svm, linear_model, preprocessing , utils
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import joblib


def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [MPG]')
    plt.legend()
    plt.grid(True)

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
            if isfile('output.csv'):
                print('found data file!')
                dataset = np.genfromtxt('output.csv', delimiter=',', dtype =np.float32)
            else:
                print('didnt found data file!')
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
            joblib.dump(scaler, 'std_scaler.bin', compress=True)
            Y = dataset[:,-1]
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=92)

            #model = svm.LinearSVR(epsilon = 0.1, C = 0.15, max_iter=2000)
            model = linear_model.SGDRegressor(max_iter=10000)
            model.fit(X_train, y_train)
            print(f'score: {model.score(X_test, y_test)}')
            joblib.dump(model, 'model_color.pkl')
            test_predictions = model.predict(X_test).flatten()

            a = plt.axes(aspect='equal')
            plt.scatter(y_test, test_predictions)
            plt.xlabel('True Values [MPG]')
            plt.ylabel('Predictions [MPG]')
            lims = [1, 5]
            plt.xlim(lims)
            plt.ylim(lims)
            plt.plot(lims, lims)
            plt.savefig('predict_col.png')
        
        else:
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
            
            X = np.array(X)
            Y = np.array(Y)
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=92)
            X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size=0.5, random_state=92)
            base_model = tf.keras.applications.MobileNetV2(include_top=False, input_shape=(image_size[0],image_size[1],3))
            base_model.trainable = False

            model2 = keras.Sequential(
                [
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dropout(0.2),
                layers.Dense(1)
                ]
                )

            model2.summary()
            model2.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.001))

            history = model2.fit(
                X_train,
                y_train,
                validation_data = (X_valid, y_valid),
                verbose=1, epochs=10
                )
            
            test_predictions = model2.predict(X_test).flatten()

            a = plt.axes(aspect='equal')
            plt.scatter(y_test, test_predictions)
            plt.xlabel('True Values [MPG]')
            plt.ylabel('Predictions [MPG]')
            lims = [1, 5]
            plt.xlim(lims)
            plt.ylim(lims)
            plt.plot(lims, lims)
            plt.savefig('predict.png')

            model2.save("model_transfer.h5")


scheduler = Scheduler()
scheduler.run(False)
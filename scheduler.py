from bgFeatureExtraction import bgExtractor
import cv2
from os import listdir
from os.path import isfile, join
import pandas as pd

class Scheduler:

    def __init__(self) -> None:
        self.db_path = 'SCUT-FBP5500_v2'

    def load_img_paths(self):
        files = [f for f in listdir(f'{self.db_path}\\Images') if isfile(join(f'{self.db_path}\\Images', f))]
        return files

    def load_img_labels(self):
        excel_df = pd.read_excel(f'{self.db_path}\\All_Ratings.xlsx', sheet_name='ALL', usecols=['Filename', 'Rating'])
        return excel_df


    def run(self):
        ratings_df = self.load_img_labels()
        ratings_df_grouped = ratings_df.groupby('Filename').mean()
        print(ratings_df_grouped)
        imgs_filenames = self.load_img_paths()
        main_colors = dict()
        for img_filename in imgs_filenames:
            img = cv2.imread(f'{self.db_path}\\Images\\{img_filename}')
            main_colors[img_filename] = bgExtractor.run(img)
            ratings_df_grouped.loc[img_filename, ['main_colors']] = main_colors[img_filename]
            print(ratings_df_grouped.loc[img_filename,:])


scheduler = Scheduler()
scheduler.run()
import pandas as pd
import numpy as np
import joblib
# from sklearn import preprocessing
# from sklearn import model_selection

def preprocess_data(data_path):    
    df = pd.read_csv(data_path)
    # convert everything to float for safety
    df['xtl'] = df['xtl'].astype(np.float)
    df['ytl'] = df['ytl'].astype(np.float)
    
    df['xbr'] = df['xbr'].astype(np.float)
    df['ybr'] = df['ybr'].astype(np.float)
    
    # We need to do some encoding for has_mask an has_helmet categorical columns
    # Faster RCNN considers 0 to be background class so we cannot encode any class as 0
    
    # Let's assign the mask classes manually right now
    # yes = 1
    # no = 2
    # invisible = 3
    # wrong = 1 (There is only one instance of this but still lets keep it)

    # print(df['has_mask'].value_counts())

    # Hard coding a bit.
    print(df.head())

    mask_dict = {
        'yes' : '1',
        'no' : '2',
        'invisible' : '3',
        'wrong' : '4'
    }

    joblib.dump(mask_dict, 'data/models/mask_dict.pkl')

    # has_helmet needs the same preprcessing too

    # print(df['has_helmet'].value_counts())

    # yes -> 1
    # no -> 2

    helmet_dict = {
        'yes': '1',
        'no': '2'
    }

    joblib.dump(helmet_dict, 'data/models/helm_dict.pkl')

    for i, val in enumerate(df['has_helmet']):
        df['has_helmet'].iloc[i] = helmet_dict[val]
    

    for i, val in enumerate(df['has_mask']):
        df['has_mask'].iloc[i] = mask_dict[val]

    print(df.head())
    
    # Additionally we can perform train_test_split here but we have small data right now
    # I'm leaving it right now

    # Part column makes no sense right now
    df.drop(["part"], axis=1, inplace=True)


    # I'm right now using a portion of same train_data as validation data.
    # Since I have very low data and just to show validation. I will avoid dividing data into two here.

    df_train = df
    df_val = df[1692:]

    # print(df_train.head())
    # print(df_val.tail())

    df_train.to_csv("data/df_train.csv", index=False)
    df_val.to_csv("data/df_val.csv", index=False)


if __name__ == "__main__":
    preprocess_data("data/face_data.csv")


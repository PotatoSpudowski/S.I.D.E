import pandas as pd
from tqdm import tqdm
from utils import utils

df = pd.read_csv("Data/GCC_data/val_data.csv")
ignore_list = []

for i in tqdm(range(len(df))):
    image = df["images"][i]
    if utils.verify_image("Data/GCC_data/"+image):
        continue
    else: 
        ignore_list.append(image)

df2 = pd.DataFrame(ignore_list)
df2.to_csv("Data/GCC_data/Ignore_list.csv")

import fasttext
import pandas as pd
import warnings
import config as cfg
warnings.filterwarnings("ignore")
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
train_data = pd.read_csv(cfg.training_data,sep="\t")
test_data = pd.read_csv(cfg.test_data, sep = "\t")

fastext_data = []

#Uncomment this if you want to generate the data in fasttext format
'''for index,row in train_data.iterrows():
    if row['Phrase'].lower() not in stop_words:
        fastext_data.append("__label__"+str(row['Sentiment'])+" "+ row['Phrase'])

print("Writing to file")
with open('/home/venky/training_data.txt', 'w') as f:
    for item in fastext_data:
        f.write("%s\n" % item)
'''
model = fasttext.train_supervised(input="/home/venky/training_data.txt")
print(f"precision is {model.test(cfg.fasttext_supervised_input)[1]}")
print(f"recall is {model.test(cfg.fasttext_supervised_input)[2]}")
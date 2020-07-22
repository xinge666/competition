import pandas as pd
import pickle
from  tqdm  import tqdm
import numpy as np
from gensim.models import Word2Vec
import gc


def get_sentence():
    #ad_user_sentence = pickle.load(open('ad_user_sentence.pkl','rb'))
    merge_data = pd.read_pickle('prepared_data/merge_data.pkl')
    user_max_min_dic = pickle.load(open('prepared_data/user_max_min_dic','rb'))
    sentence_dict = defaultdict(list)
    for key,value in tqdm(user_max_min_dic.items()):
        temp_df = merge_data.iloc[value['min']:value['max']+1,:]
        for col in [ 'creative_id', 'ad_id', 'product_id',
        'product_category', 'advertiser_id', 'industry']:
            sentence_dict[col].append(temp_df[col].tolist())
    pickle.dump(sentence_dict,open('prepared_data/sentence_dict.pkl','wb'))
    return sentence_dict

def quert_sent(value):
    temp_sentenct = []
    for key in value:
        temp_sentenct.append([str(x) for x in key])
    return temp_sentenct

sentence_dict = get_sentence() 
for key,value in sentence_dict.items():
    print(key)
    temp_sentenct = quert_sent(value)
    model = Word2Vec(temp_sentenct, size=256, window=20, min_count=5, workers=-1)
    model.save("word2vec_model/word2vec_{}.model".format(key))
    del temp_sentenct,model
    gc.collect()

!zip -r word2vec_model_256.zip word2vec_model
upload(word2vec_model_256.zip)

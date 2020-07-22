import pandas as pd
import pickle
from  tqdm  import tqdm
import numpy as np
from gensim.models import Word2Vec
import gc
import os 
from cos import download,upload
from collections import defaultdict
#ad_user_sentence = pickle.load(open('ad_user_sentence.pkl','rb'))
merge_data = pd.read_pickle('prepared_data/merge_data.pkl')
user_max_min_dic = pickle.load(open('prepared_data/user_max_min_dic','rb'))
user_df = pd.read_csv('raw_data/user.csv')
age = user_df.age
gender = user_df.gender
user_id_2_data_dic = {}
for key,value in tqdm(user_max_min_dic.items()):
    temp_df = merge_data.iloc[value['min']:value['max']+1,:]
    user_id_2_data_dic[key]={"data":temp_df}
    if key<=3000000:
        user_id_2_data_dic[key]['age'] =age[key-1]
        user_id_2_data_dic[key]['gender'] =gender[key-1]
pickle.dump(user_id_2_data_dic,open('user_id_2_data_dic.pkl','wb'))
from cos import download,upload
upload('user_id_2_data_dic.pkl')

!pip install gensim
import pandas as pd
import pickle
from  tqdm  import tqdm
import numpy as np
from gensim.models import Word2Vec
import gc
import os 
from cos import download,upload
from collections import defaultdict
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
import pickle
pickle.dump(sentence_dict,open('id_sentence_dict.pkl','wb'))
from cos import download,upload
upload('id_sentence_dict.pkl')
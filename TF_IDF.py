# -*- coding: utf-8 -*-
# %%
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import numpy as np
import pandas as pd


np.random.seed(10)

#所有的中文
chn_pattern ="[\u4e00-\u9fa5]+"
regex_chn = re.compile(chn_pattern)

c_path = './email_2017.txt'
b_path = './fake_email.txt'

# %%
def parser(path,security_group):
    line_vec = []
    group=[]
    with open(path,'r',encoding='utf8') as f:
        f = f.readlines()
        for line in f:
            tmp = regex_chn.findall(line)
            if not tmp:
                continue
            line_vec.append(tmp)
            group.append(security_group)
    return np.asarray(line_vec),group

# %%
c_vec,c_group = parser(c_path,'C')
b_vec,b_group = parser(b_path,'B')

all_vec = np.r_[c_vec,b_vec]

print('Security_B size:{}\nSecurity_C size:{}'.format(b_vec.shape[0], c_vec.shape[0]))


# %%
def test():
    b_vec_index = list(range(0, b_vec.shape[0]))
    c_vec_index = list(range(b_vec.shape[0], b_vec.shape[0]+c_vec.shape[0]))
    
    
    b_array = np.full(b_vec.shape[0], 'B')
    c_array = np.full(c_vec.shape[0], 'C')
    security_list = np.append(b_array, c_array)
    
    
    #security_list = ['B' for _ in range(b_vec.shape[0])]
    #for _ in range(b_vec.shape[0], b_vec.shape[0]+c_vec.shape[0]):
    #    security_list.append('C') 

# %%
# TF-IDF
new_vec = []
for item in all_vec:
    string = " ".join(item)
    new_vec.append(string)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(new_vec)

df = pd.DataFrame(X.toarray()) 
df.insert(0, "security", security_list) 
# 維度
# (5184, 8651)
# %%
df.to_csv('email_dataset.csv', index = False)

# %%

%time test()

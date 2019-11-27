from sklearn.feature_extraction.text import TfidfVectorizer
import re
import numpy as np
import pandas as pd


np.random.seed(10)

chn_pattern ="[\u4e00-\u9fa5]+"
regex_chn = re.compile(chn_pattern)

c_path = './email_2017.txt'
b_path = './fake_email.txt'

# %%
def parser(path,security_group):
    line_vec = [];group=[]
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

print('Security_B size:{}, Security_C size:{}'.format(b_vec.shape[0], c_vec.shape[0]))


# %%

b_vec_index = list(range(0, b_vec.shape[0]))
c_vec_index = list(range(b_vec.shape[0], b_vec.shape[0]+c_vec.shape[0]))
security_list = []
for i in range (0,len(b_vec_index)):
    security_list.append('B')



len(security_list)
    
# %%

# %%    
# TF-IDF
new_vec = []
for item in all_vec:
    string = " ".join(item)
    new_vec.append(string)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(new_vec)

df = pd.DataFrame(X.toarray()) 
df.insert(0, "security", ) 
# 維度
# (5184, 8651)
# %%



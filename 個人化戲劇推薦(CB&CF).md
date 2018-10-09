
# 個人化戲劇推薦系統的建立

### 此檔案主要是利用「基於內容」推薦算法(CB)及「協同過濾」推薦算法(CF)，計算每個使用者對每部戲劇的個人喜好分數，再搭配三個推薦策略，得到「個人化+熱門度+最新流行」的推薦結果

## 1. 資料準備

### 1.1 匯入所需要的套件及資料


```python
import pandas as pd
import numpy as np
import math
import scipy
import sys
import random

from scipy import spatial
from collections import Counter
from operator import itemgetter
from collections import defaultdict

tbl_DramaGenre=pd.read_csv('tbl_DramaGenre.csv', encoding='utf-8') #戲劇類型對照表
tbl_Genre=pd.read_csv('tbl_Genre.csv', encoding='utf-8')  #類型基本資料
tbl_Drama=pd.read_csv('tbl_Drama.csv', encoding='utf-8')  #戲劇基本資料
tbl_Actor=pd.read_csv('tbl_Actor.csv', encoding='utf-8')  #演員基本資料
tbl_DramaActor=pd.read_csv('tbl_DramaActor.csv', encoding='utf-8') #戲劇演員對照表
df_drama_ott=pd.read_csv('df_drama_ott.csv', encoding='utf-8') #戲劇OTT平台對照表
df_Drama_GenreName=pd.read_csv('df_Drama_GenreName.csv', encoding='utf-8') #戲劇類型對照表
random_result=pd.read_csv('random_result_5000_v3.csv', encoding='utf-8') #5000筆模擬資料,包括ip、dramaID、收藏日期、收藏次數
random_result2=pd.read_csv('random_result_5000_v2.csv', encoding='utf-8') #5000筆模擬資料,包括ip、dramaID、收藏日期、收藏次數
df_drama_zone_genre_actor_all=pd.read_csv('df_drama_zone_genre_actor_all2.csv', encoding='utf-8') #每部片的地區、類型、演員特徵
```

### 1.2 將5000筆模擬資料整理成dictionary格式，俾利後續CF(user-based)及LFM演算法的計算


```python
def data_processing(histroryDrama, user_ip):
    
    #將5000個使用者的個人ip及dramaID的對應關係整理成dictionary形式
    ip_history_list=list(random_result['ipAddr'].tolist())

    UserItemDict={}
    for i in ip_history_list:
        ll=random_result[random_result['ipAddr']==i]['dramaID'].tolist()
        UserItemDict[i]=ll[0].split("/")
    
    #取得新使用者的收藏片單並且整理成list
    histroryDrama_list=[]
    for i in histroryDrama:
        histroryDrama_list.append(i[1])

    #將新的使用者的看片清單加入5000人的消費者使用清單中(不將新的使用者資料併入5000筆計算基礎中使用)
    UserItemDict[user_ip]=histroryDrama_list
        
    UserItemDictPositiveGrade={}

    if user_ip in ip_history_list:
        ip_history_list_new=ip_history_list
    else:
        ip_history_list_new=ip_history_list+[user_ip]

    for i in ip_history_list_new:
        drama_dic1={}
        for j in UserItemDict[i]:
            drama_dic1[j]=1
            UserItemDictPositiveGrade[i]=drama_dic1
    
    #取得所有戲劇的劇名ID清單
    dramaName_list=list(set(df_drama_zone_genre_actor_all['dramaID']))
    dramaName_list2=list(set(random_result2['dramaID'].tolist()))
    
    
    # 將未觀看過的戲劇加入user-item的字典中並設定對應值為零，以做為計算LFM的基礎
    UserItemDictGrade={}
    for user in list(UserItemDict.keys()):
        itemDict={}
        positiveItemList=UserItemDict[user]
        otherItemList=list(set(dramaName_list2)-set(positiveItemList))
        negativeItemList = random.sample(otherItemList, len(positiveItemList))
        for i in positiveItemList:
            itemDict[i]=1
        for j in negativeItemList:
            itemDict[j]=0
        UserItemDictGrade[user]=itemDict
    
    return UserItemDict, UserItemDictPositiveGrade, UserItemDictGrade, dramaName_list, dramaName_list2
```


```python
histroryDrama=[('127.0.0.1', 'G020073'), ('127.0.0.1', 'G010012'), ('127.0.0.1', 'G010003'), ('127.0.0.1', 'G010002'), \
                  ('127.0.0.1', 'G010005'),('127.0.0.1', 'G010056'), ('127.0.0.1', 'G010169'), ('127.0.0.1', 'G020041') ]

UserItemDict, UserItemDictPositiveGrade, UserItemDictGrade, dramaName_list, dramaName_list2=data_processing(histroryDrama, '127.0.0.1')
```

## 2. 「基於內容演算法」(Content-Based Recommendations)的計算

### 2.1 根據使用者的收藏片單資料，整理出其對應的戲劇特徵矩陣(one-hot encoding)


```python
def drama_feature_ind(histroryDrama, user_ip):
    
    UserItemDict, UserItemDictPositiveGrade, UserItemDictGrade, dramaName_list, dramaName_list2=data_processing(histroryDrama, user_ip)
    df_drama_zone_genre_actor_ind=pd.DataFrame()

    for i in UserItemDict[user_ip]:
        df_drama_feature=df_drama_zone_genre_actor_all[df_drama_zone_genre_actor_all['dramaID']==i]
        df_drama_zone_genre_actor_ind=df_drama_zone_genre_actor_ind.append(df_drama_feature)

    return df_drama_zone_genre_actor_ind
```

### 2.2 根據使用者的收藏片單資料，計算使用者對不同戲劇特徵的喜好度


```python
def user_feature_grade(histroryDrama, user_ip):
    
    #首先從使用者收藏清單中，計算各個戲劇特徵的出現次數，並計算該次數佔總收藏片數的比重
    
    df_drama_zone_genre_actor_ind = drama_feature_ind(histroryDrama, user_ip)
    df_ind=list(df_drama_zone_genre_actor_ind.mean(axis=0))[0:]

    #針對每個戲劇特徵進行比重的調整

    df_ind1=[]
    for i in df_ind[0:3]:
        df_ind1.append(i/3) #降低地區特徵的重要性
    for i in df_ind[3:15]:
         df_ind1.append(i)
    for i in df_ind[15:16]:
        df_ind1.append(i/2) #降低戲劇類型中「劇情」的重要性
    for i in df_ind[16:22]:
         df_ind1.append(i)
    for i in df_ind[22:23]:
        df_ind1.append(i/2) #降低戲劇類型中「愛情」的重要性
    for i in df_ind[23:168]:
        df_ind1.append(i)   #使用者對每個戲劇特徵的喜好分數
    
    return df_ind1 
```

### 2.3 根據每個使用者對不同戲劇特徵的喜好度，利用餘弦相似度計算特定使用者對不同戲劇的喜好程度


```python
def recommend_CB(histroryDrama, user_ip):
    
    UserItemDict, UserItemDictPositiveGrade, UserItemDictGrade, dramaName_list, dramaName_list2=data_processing(histroryDrama, user_ip)
    df_ind1=user_feature_grade(histroryDrama, user_ip)

    rank_CB={}
    for i in dramaName_list:
        if i in UserItemDict[user_ip]:
            pass
        else:
            df_ind_drama=np.array(df_drama_zone_genre_actor_all[df_drama_zone_genre_actor_all['dramaID']==i]).tolist()
            df_ind_drama1=df_ind_drama[0][1:-1] 
            sum_ab=0
            sum_a2=0
            sum_b2=0
            for j in range(0,168,1):
                sum_ab=sum_ab+df_ind_drama1[j]*df_ind1[j] #ind_drama1為每部戲劇所對應的特徵, df_ind1為使用者對每個戲劇特徵的喜好分數
                sum_a2=sum_a2+(df_ind_drama1[j])**2
                sum_b2=sum_b2+(df_ind1[j])**2        
            rank_CB[i]=round(sum_ab/((sum_a2**0.5)*(sum_b2**0.5)),4)

    rank_sorted_CB=sorted(rank_CB.items(), key=lambda d: d[1], reverse=True)
    
    return rank_sorted_CB
```


```python
histroryDrama=[('127.0.0.1', 'G020073'), ('127.0.0.1', 'G010012'), ('127.0.0.1', 'G010003'), ('127.0.0.1', 'G010002'), \
                  ('127.0.0.1', 'G010005'),('127.0.0.1', 'G010056'), ('127.0.0.1', 'G010169'), ('127.0.0.1', 'G020041') ]

grade_CB=recommend_CB(histroryDrama, '127.0.0.1')

CB={}
for i in grade_CB:
    CB[i[0]]=i[1]
    
CB['G010166'] #根據CB演算法，使用者對G010075(dramaID)的可能評分
```




    0.5192



## 3.「協同過濾演算法」(User-baded CollaboratIve Filtering)的計算

### 3.1建立物品到用户的倒排表


```python
def movie_users_inverse_table(data):

    # build inverse table for item-users
    # key=movieID, value=list of userIDs who have seen this movie

    movie_users = dict()
    movie_popular = dict()

    for user, movies in data.items():
        for movie in movies:

            # inverse table for item-users
            if movie not in movie_users:
                movie_users[movie] = set()
            movie_users[movie].add(user)

            # count item popularity at the same time
            if movie not in movie_popular:
                movie_popular[movie] = 0
            movie_popular[movie] += 1
    print ('build movie-users inverse table succ', file=sys.stderr)

    return movie_users, movie_popular
```

### 3.2計算使用者共同收藏的戲劇個數


```python
def count_corated_items():
    
    # count co-rated items between users
    # key=userID value=dict of otherUserID with count of co-rated items
    movie_users, movie_popular = movie_users_inverse_table(data)
    
    users_corated_count=dict()

    for movie, users in movie_users.items():
        for u in users:
            users_corated_count.setdefault(u, defaultdict(int))
            for v in users:
                if u == v:
                    continue
                users_corated_count[u][v] += 1
    print ('build user co-rated movies matrix succ', file=sys.stderr)

    return users_corated_count
```

### 3.3建立使用者相似度矩陣


```python
def similarity_matrix(data):
   
    # calculate similarity matrix
    
    users_corated_count = count_corated_items()

    simMatrix=dict()

    simfactor_count = 0
    PRINT_STEP = 2000000

    for u, related_users in users_corated_count.items():
        simMatrix.setdefault(u, defaultdict(int))
        for v, count in related_users.items():
            simMatrix[u][v] = count / math.sqrt(
                len(data[u]) * len(data[v]))
            simfactor_count += 1
            if simfactor_count % PRINT_STEP == 0:
                print ('calculating user similarity factor(%d)' % simfactor_count, file=sys.stderr)

    print ('calculate user similarity matrix(similarity factor) succ', file=sys.stderr)
    print ('Total similarity factor number = %d' % simfactor_count, file=sys.stderr)

    return simMatrix
```

### 3.4 針對特定使用者，找出喜好最相近的前20人，計算其對每個戲劇的喜好程度


```python
def recommend_CF_userBased(user_ip, data):
    
    simMatrix = similarity_matrix(data)
    
    simMatrix_list=list(simMatrix[user_ip].items())
    simMatrix_list.sort(key=lambda x: (x[1], x[0]), reverse=True)
    watched_movies = data[user_ip]
    
    rank_CF=dict()
    for i in simMatrix_list[0:20]:
        for movie in data[i[0]]:
            if movie in watched_movies:
                continue
            if movie not in rank_CF:
                rank_CF.setdefault(movie, 0)
            rank_CF[movie] += i[1]

    rank_sorted_CF=sorted(rank_CF.items(), key=itemgetter(1), reverse=True)

    return rank_sorted_CF
```


```python
histroryDrama=[('127.0.0.1', 'G020073'), ('127.0.0.1', 'G010012'), ('127.0.0.1', 'G010003'), ('127.0.0.1', 'G010002'), \
                  ('127.0.0.1', 'G010005'),('127.0.0.1', 'G010056'), ('127.0.0.1', 'G010169'), ('127.0.0.1', 'G020041') ]

UserItemDict, UserItemDictPositiveGrade, UserItemDictGrade, dramaName_list, dramaName_list2=data_processing(histroryDrama,'127.0.0.1')
data=UserItemDictPositiveGrade

grade_CF=recommend_CF_userBased('127.0.0.1', data)

CF={}
for i in grade_CF:
    CF[i[0]]=i[1]
    
CF['G010166'] #根據CF演算法，使用者對G010075(dramaID)的可能評分
```

    build movie-users inverse table succ
    build user co-rated movies matrix succ
    calculating user similarity factor(2000000)
    calculating user similarity factor(4000000)
    calculating user similarity factor(6000000)
    calculate user similarity matrix(similarity factor) succ
    Total similarity factor number = 7199428
    




    1.118033988749895



## 4.「協同過濾演算法」(Item-baded CollaboratIve Filtering)的計算

### 4.1建立物品的熱門度表


```python
def calc_movie_popular(data):
    
    movie_popular={}
    movie_count=0
    
    # count movie popularity
    for user, movies in data.items():
        for movie in movies:
            if movie not in movie_popular:
                movie_popular[movie] = 0
            movie_popular[movie] += 1

    print('count movies number and popularity succ', file=sys.stderr)

    # save the total number of movies
    movie_count = len(movie_popular)
#     print('total movie number = %d' % movie_count, file=sys.stderr)
    
    return movie_popular, movie_count
```

### 4.2 計算戲劇共同被相同使用者觀看的次數


```python
def count_corated_users(data):

    # count co-rated users between items
    # key=movieID value=dict of otherMovieID with count of co-rated users
    
    items_coratedUser_count = {}

    for user, movies in data.items():
        for m1 in movies:
            items_coratedUser_count.setdefault(m1, defaultdict(int))
            for m2 in movies:
                if m1 == m2:
                    continue
                items_coratedUser_count[m1][m2] += 1

    print('build item co-rated users matrix succ', file=sys.stderr)
    
    return items_coratedUser_count
```

### 4.3建立戲劇相似度矩陣


```python
def similarity_matrix_I():
    
    # calculate similarity matrix
    
    movie_popular, movie_count=calc_movie_popular(data)
    items_coratedUser_count=count_corated_users(data)
    
    simMatrix_I={}
    
    simfactor_count = 0
    PRINT_STEP = 10000000

    for m1, related_movies in items_coratedUser_count.items():
        simMatrix_I.setdefault(m1, defaultdict(int))
        for m2, count in related_movies.items():
            simMatrix_I[m1][m2] = count / math.sqrt(movie_popular[m1] * movie_popular[m2])
            simfactor_count += 1
            if simfactor_count % PRINT_STEP == 0:
                print('calculating movie similarity factor(%d)' % simfactor_count, file=sys.stderr)

    print('calculate movie similarity matrix(similarity factor) succ', file=sys.stderr)
#     print('Total similarity factor number = %d' % simfactor_count, file=sys.stderr)
    
    return simMatrix_I
```

### 4.4 針對特定戲劇，找出最相近的前20部戲劇，計算使用者對每部戲劇的喜好程度


```python
def recommend_CF_itemBased(user, data):
    
    simMatrix_I=similarity_matrix_I()
    
    rank_I = {}
    watched_movies = data[user]

    for movie, rating in watched_movies.items():
        for related_movie, similarity_factor in sorted(simMatrix_I[movie].items(),key=itemgetter(1), reverse=True)[:20]:
            if related_movie in watched_movies:
                continue
            rank_I.setdefault(related_movie, 0)
            rank_I[related_movie] += similarity_factor
    # return the N best movies
    return sorted(rank_I.items(), key=itemgetter(1), reverse=True)
```


```python
histroryDrama=[('127.0.0.1', 'G020073'), ('127.0.0.1', 'G010012'), ('127.0.0.1', 'G010003'), ('127.0.0.1', 'G010002'), \
                  ('127.0.0.1', 'G010005'),('127.0.0.1', 'G010056'), ('127.0.0.1', 'G010169'), ('127.0.0.1', 'G020041') ]

UserItemDict, UserItemDictPositiveGrade, UserItemDictGrade, dramaName_list, dramaName_list2=data_processing(histroryDrama,'127.0.0.1')
data=UserItemDictPositiveGrade

grade_CF_I=recommend_CF_itemBased('127.0.0.1', data)

CF_I={}
for i in grade_CF_I:
    CF_I[i[0]]=i[1]
    
CF_I['G010166'] #根據CF演算法，使用者對G010075(dramaID)的可能評分
```

    count movies number and popularity succ
    build item co-rated users matrix succ
    calculate movie similarity matrix(similarity factor) succ
    




    0.3353389491437977



## 5. 對CB及CF得到的分數進行加權平均，得到每個使用者的個人喜好分數


```python
def aveGrade():
    
    mix={}
    for key in CB:
        try:
            mix[key]=(CB[key]+CF[key])/2 
        except:
            mix[key]=CB[key]
    mix2={}
    for key in CB:
        try:
            mix2[key]=(mix[key]+CF_I[key])/2
        except:
            mix2[key]=mix[key]

    r_mix=sorted(mix2.items(), key=lambda x:-x[1])

    r_mix_dic={}
    for i in r_mix:
        r_mix_dic[i[0]]=i[1]
        
    return r_mix_dic
```


```python
r_mix_dic=aveGrade()
r_mix_dic['G010166'] #CB及CF加權後， 使用者對G010075(dramaID)的喜好分數
```




    0.5769779717593726




```python
r_mix_dic_sorted=sorted(r_mix_dic.items(), key=itemgetter(1), reverse=True)
r_mix_dic_sorted
```




    [('G030121', 0.6982310457736804),
     ('G010011', 0.692530265547432),
     ('G010162', 0.6882112859634918),
     ('G030140', 0.6862768429493377),
     ('G010136', 0.6852),
     ('G010103', 0.6675),
     ('G010025', 0.6675),
     ('G010109', 0.6675),
     ('G010264', 0.6634),
     ('G030092', 0.6592865978039979),
     ('G010080', 0.6294),
     ('G010041', 0.6294),
     ('G010165', 0.6294),
     ('G010064', 0.6294),
     ('G010034', 0.6294),
     ('G010072', 0.6294),
     ('G010035', 0.6294),
     ('G010090', 0.6294),
     ('G010163', 0.6294),
     ('G010192', 0.6294),
     ('G010100', 0.6294),
     ('G010106', 0.6294),
     ('G010019', 0.6294),
     ('G010031', 0.6294),
     ('G010168', 0.6294),
     ('G010070', 0.6294),
     ('G010295', 0.6294),
     ('G010071', 0.6294),
     ('G010023', 0.6294),
     ('G010294', 0.6294),
     ('G010076', 0.6294),
     ('G010267', 0.6294),
     ('G010107', 0.6294),
     ('G010141', 0.6294),
     ('G010124', 0.6294),
     ('G010085', 0.6294),
     ('G010186', 0.6294),
     ('G010142', 0.6294),
     ('G010054', 0.6294),
     ('G010207', 0.6294),
     ('G010036', 0.6294),
     ('G010078', 0.6294),
     ('G010050', 0.6294),
     ('G010098', 0.6294),
     ('G010039', 0.6294),
     ('G030196', 0.6234380139940858),
     ('G030082', 0.6000528129762285),
     ('G010304', 0.5995),
     ('G010063', 0.5995),
     ('G010308', 0.5995),
     ('G010093', 0.5995),
     ('G010203', 0.5995),
     ('G010104', 0.5995),
     ('G010122', 0.5995),
     ('G010283', 0.5934),
     ('G010116', 0.5934),
     ('G010239', 0.5934),
     ('G010166', 0.5769779717593726),
     ('G010204', 0.53627432728696),
     ('G010088', 0.5307),
     ('G010055', 0.5307),
     ('G010215', 0.5307),
     ('G010115', 0.5192),
     ('G010013', 0.5192),
     ('G010152', 0.5192),
     ('G010082', 0.5192),
     ('G010234', 0.5192),
     ('G010132', 0.5192),
     ('G010183', 0.5192),
     ('G010043', 0.5192),
     ('G010089', 0.5192),
     ('G010026', 0.5192),
     ('G010062', 0.5192),
     ('G010018', 0.5192),
     ('G010246', 0.5192),
     ('G010307', 0.5192),
     ('G010067', 0.5192),
     ('G010060', 0.5139),
     ('G010051', 0.5139),
     ('G010118', 0.5139),
     ('G010015', 0.5139),
     ('G010296', 0.5139),
     ('G010046', 0.5139),
     ('G010086', 0.5139),
     ('G010084', 0.5139),
     ('G010110', 0.5139),
     ('G010074', 0.5139),
     ('G010129', 0.5139),
     ('G010143', 0.5139),
     ('G010260', 0.5139),
     ('G010101', 0.5139),
     ('G010171', 0.5139),
     ('G010188', 0.5139),
     ('G010144', 0.5139),
     ('G010038', 0.5139),
     ('G010073', 0.5139),
     ('G010226', 0.5139),
     ('G010259', 0.5139),
     ('G010121', 0.5139),
     ('G010044', 0.5139),
     ('G010096', 0.5139),
     ('G010251', 0.5139),
     ('G010245', 0.5139),
     ('G010167', 0.5139),
     ('G010256', 0.5139),
     ('G010235', 0.5139),
     ('G010017', 0.5139),
     ('G010102', 0.5139),
     ('G010224', 0.5139),
     ('G010195', 0.5139),
     ('G010094', 0.5139),
     ('G010105', 0.5139),
     ('G010190', 0.5139),
     ('G010123', 0.5139),
     ('G010208', 0.5046),
     ('G020048', 0.4945),
     ('G020042', 0.4895),
     ('G020030', 0.4895),
     ('G020062', 0.4895),
     ('G020044', 0.4895),
     ('G020123', 0.4895),
     ('G020129', 0.4895),
     ('G020040', 0.4895),
     ('G020119', 0.4895),
     ('G020099', 0.4895),
     ('G020122', 0.4895),
     ('G020080', 0.4895),
     ('G020020', 0.4895),
     ('G020028', 0.4895),
     ('G020112', 0.4895),
     ('G020093', 0.4895),
     ('G020063', 0.4895),
     ('G020115', 0.4895),
     ('G020032', 0.4895),
     ('G020017', 0.4895),
     ('G020016', 0.4895),
     ('G020047', 0.4895),
     ('G020038', 0.4895),
     ('G020089', 0.4895),
     ('G020084', 0.4895),
     ('G020029', 0.4895),
     ('G020101', 0.4895),
     ('G020114', 0.4895),
     ('G020124', 0.4895),
     ('G020086', 0.4895),
     ('G020046', 0.4895),
     ('G020055', 0.4895),
     ('G020125', 0.4895),
     ('G020104', 0.4895),
     ('G020113', 0.4895),
     ('G020109', 0.4853),
     ('G020059', 0.4853),
     ('G020128', 0.4853),
     ('G020107', 0.4853),
     ('G020120', 0.4853),
     ('G020079', 0.4853),
     ('G010146', 0.4845),
     ('G010004', 0.483206797749979),
     ('G010211', 0.4644),
     ('G010138', 0.4644),
     ('G010212', 0.4644),
     ('G010170', 0.4644),
     ('G010178', 0.4644),
     ('G010189', 0.4644),
     ('G010299', 0.4644),
     ('G010305', 0.4555008043791047),
     ('G010293', 0.45353324155736957),
     ('G010081', 0.4485),
     ('G010222', 0.4485),
     ('G010292', 0.4485),
     ('G010127', 0.445),
     ('G010092', 0.445),
     ('G010030', 0.445),
     ('G010197', 0.445),
     ('G010114', 0.445),
     ('G010276', 0.445),
     ('G010236', 0.445),
     ('G010232', 0.445),
     ('G010216', 0.445),
     ('G010120', 0.445),
     ('G010269', 0.445),
     ('G010135', 0.445),
     ('G010047', 0.445),
     ('G010045', 0.445),
     ('G010061', 0.445),
     ('G010020', 0.445),
     ('G010099', 0.445),
     ('G010200', 0.445),
     ('G010298', 0.445),
     ('G010277', 0.445),
     ('G010077', 0.445),
     ('G030111', 0.4282),
     ('G030094', 0.4282),
     ('G030325', 0.4282),
     ('G030207', 0.4282),
     ('G030167', 0.4282),
     ('G030218', 0.4282),
     ('G030134', 0.4282),
     ('G030257', 0.4282),
     ('G010247', 0.42650339887498945),
     ('G010052', 0.4239),
     ('G020067', 0.42345679774997896),
     ('G010153', 0.422606797749979),
     ('G010049', 0.42200520999743063),
     ('G020056', 0.4203),
     ('G020074', 0.4203),
     ('G030056', 0.4196),
     ('G030307', 0.4196),
     ('G030128', 0.4196),
     ('G030357', 0.4196),
     ('G030077', 0.4196),
     ('G030280', 0.4196),
     ('G030156', 0.4196),
     ('G030120', 0.4196),
     ('G030296', 0.4196),
     ('G030251', 0.4196),
     ('G030329', 0.4196),
     ('G030059', 0.4196),
     ('G030173', 0.4196),
     ('G030288', 0.4196),
     ('G030091', 0.4196),
     ('G030090', 0.4196),
     ('G030015', 0.4196),
     ('G030103', 0.4196),
     ('G030299', 0.4196),
     ('G030033', 0.4196),
     ('G030172', 0.4196),
     ('G030186', 0.4196),
     ('G030303', 0.4196),
     ('G030356', 0.4196),
     ('G030110', 0.4196),
     ('G030144', 0.4196),
     ('G030139', 0.4196),
     ('G030135', 0.4196),
     ('G030093', 0.4196),
     ('G030132', 0.4196),
     ('G030311', 0.4196),
     ('G030252', 0.4196),
     ('G030141', 0.4196),
     ('G030315', 0.4196),
     ('G030247', 0.4196),
     ('G030256', 0.4196),
     ('G030291', 0.4196),
     ('G030137', 0.4196),
     ('G030085', 0.4196),
     ('G030157', 0.4196),
     ('G030342', 0.4196),
     ('G030185', 0.4196),
     ('G030019', 0.4196),
     ('G030182', 0.4196),
     ('G030234', 0.4196),
     ('G030065', 0.4196),
     ('G030189', 0.4196),
     ('G030114', 0.4196),
     ('G030309', 0.4196),
     ('G030096', 0.4196),
     ('G030117', 0.4196),
     ('G030153', 0.4196),
     ('G030099', 0.4196),
     ('G030302', 0.4196),
     ('G030106', 0.4196),
     ('G030181', 0.4196),
     ('G030286', 0.4196),
     ('G030164', 0.4196),
     ('G030358', 0.4196),
     ('G030212', 0.4196),
     ('G030004', 0.4196),
     ('G030323', 0.4196),
     ('G030223', 0.4196),
     ('G030236', 0.4196),
     ('G030170', 0.4196),
     ('G010075', 0.41831374160519064),
     ('G010007', 0.41155524166511925),
     ('G010057', 0.4113745827667467),
     ('G010095', 0.40386296044768827),
     ('G010198', 0.40095385892632773),
     ('G020111', 0.3997),
     ('G020076', 0.3997),
     ('G020110', 0.3997),
     ('G020069', 0.3997),
     ('G020132', 0.3997),
     ('G020088', 0.3997),
     ('G020043', 0.3997),
     ('G020131', 0.3997),
     ('G020025', 0.3997),
     ('G010069', 0.3980333333333333),
     ('G010223', 0.398),
     ('G010140', 0.398),
     ('G010042', 0.398),
     ('G030036', 0.398),
     ('G010126', 0.398),
     ('G010270', 0.398),
     ('G010065', 0.398),
     ('G010091', 0.3936310363079829),
     ('G010210', 0.3925),
     ('G010219', 0.38604944487358056),
     ('G010265', 0.3792497224367902),
     ('G010221', 0.37715339887498944),
     ('G020098', 0.3759),
     ('G030133', 0.3757766952966369),
     ('G010066', 0.37140339887498947),
     ('G010206', 0.37140339887498947),
     ('G030312', 0.3709),
     ('G030098', 0.3709),
     ('G030166', 0.3709),
     ('G030155', 0.3709),
     ('G030163', 0.3709),
     ('G030131', 0.3709),
     ('G030012', 0.3709),
     ('G010254', 0.36925447255899807),
     ('G010191', 0.3687533988749895),
     ('G010128', 0.3680001790889026),
     ('G010119', 0.3660975303893992),
     ('G010097', 0.3628125224324688),
     ('G010133', 0.3603435464587638),
     ('G010112', 0.3598884779324264),
     ('G010181', 0.35821941398892443),
     ('G010048', 0.35552482904638627),
     ('G020095', 0.3544533988749895),
     ('G020027', 0.3544533988749895),
     ('G010125', 0.35353735771539274),
     ('G010249', 0.352056797749979),
     ('G020031', 0.3461),
     ('G020081', 0.3461),
     ('G020036', 0.3461),
     ('G010137', 0.34568152121746254),
     ('G010008', 0.345189978529317),
     ('G010289', 0.3440033988749895),
     ('G030219', 0.3426),
     ('G030105', 0.3426),
     ('G030297', 0.3426),
     ('G030242', 0.3426),
     ('G030316', 0.3426),
     ('G030210', 0.3426),
     ('G030209', 0.3426),
     ('G030188', 0.3426),
     ('G030282', 0.3426),
     ('G030060', 0.3426),
     ('G010177', 0.3426),
     ('G030050', 0.3426),
     ('G030122', 0.3426),
     ('G030052', 0.3426),
     ('G030165', 0.3426),
     ('G030039', 0.3426),
     ('G030115', 0.3426),
     ('G030277', 0.3426),
     ('G030001', 0.3426),
     ('G030205', 0.3426),
     ('G030159', 0.3426),
     ('G030287', 0.3426),
     ('G030275', 0.3426),
     ('G030057', 0.3426),
     ('G030062', 0.3426),
     ('G030112', 0.3426),
     ('G030169', 0.3426),
     ('G030292', 0.3426),
     ('G030270', 0.3426),
     ('G030042', 0.3426),
     ('G030276', 0.3426),
     ('G030298', 0.3426),
     ('G030348', 0.3426),
     ('G030238', 0.3426),
     ('G030061', 0.3426),
     ('G030058', 0.3426),
     ('G030353', 0.3426),
     ('G030225', 0.3426),
     ('G030160', 0.3426),
     ('G010306', 0.337039685923002),
     ('G030239', 0.33663177597059657),
     ('G010272', 0.33485679774997895),
     ('G010176', 0.3343033988749895),
     ('G010271', 0.3343033988749895),
     ('G010214', 0.3343033988749895),
     ('G010022', 0.3343033988749895),
     ('G030179', 0.3317),
     ('G030075', 0.3317),
     ('G030193', 0.3317),
     ('G030195', 0.3317),
     ('G030354', 0.3317),
     ('G030272', 0.3317),
     ('G030273', 0.32160339887498945),
     ('G030158', 0.32160339887498945),
     ('G030100', 0.32160339887498945),
     ('G010131', 0.31814354645876386),
     ('G010290', 0.3147),
     ('G010228', 0.3147),
     ('G010029', 0.3147),
     ('G010179', 0.3147),
     ('G010268', 0.3147),
     ('G010130', 0.3146850269189626),
     ('G020102', 0.3116533988749895),
     ('G020064', 0.31100679774997897),
     ('G010209', 0.3080533988749895),
     ('G010175', 0.30727312946227964),
     ('G030074', 0.3028),
     ('G010202', 0.3024045454545455),
     ('G010205', 0.3015569415042095),
     ('G010266', 0.30080290096535145),
     ('G010150', 0.2999520725966341),
     ('G010281', 0.2986166666666667),
     ('G030149', 0.29725339887498947),
     ('G010194', 0.2968044725589981),
     ('G030044', 0.2967),
     ('G010261', 0.2967),
     ('G030240', 0.2967),
     ('G010187', 0.2967),
     ('G030322', 0.2967),
     ('G030249', 0.2967),
     ('G030220', 0.2967),
     ('G030331', 0.2967),
     ('G030051', 0.2967),
     ('G030266', 0.2967),
     ('G030031', 0.2967),
     ('G030125', 0.2967),
     ('G030054', 0.2967),
     ('G030228', 0.2967),
     ('G030313', 0.2967),
     ('G010083', 0.2967),
     ('G030206', 0.2967),
     ('G030346', 0.2967),
     ('G030089', 0.2967),
     ('G030301', 0.2967),
     ('G010213', 0.2967),
     ('G030084', 0.2967),
     ('G030066', 0.2967),
     ('G030224', 0.2967),
     ('G030176', 0.2967),
     ('G030007', 0.2967),
     ('G030254', 0.2967),
     ('G030071', 0.2967),
     ('G030243', 0.2967),
     ('G030201', 0.2967),
     ('G010282', 0.2967),
     ('G030147', 0.2967),
     ('G010302', 0.2967),
     ('G030138', 0.2967),
     ('G030078', 0.2967),
     ('G030271', 0.2967),
     ('G030324', 0.2967),
     ('G030097', 0.2967),
     ('G030127', 0.2967),
     ('G030102', 0.29358474935575285),
     ('G020054', 0.29354500364742664),
     ('G010087', 0.28993502691896256),
     ('G030108', 0.2888569415042095),
     ('G020052', 0.28877254531628116),
     ('G020116', 0.28826941398892447),
     ('G010079', 0.28793485830098975),
     ('G020057', 0.2875966574975495),
     ('G020090', 0.28685759605332595),
     ('G020049', 0.285662083766475),
     ('G010250', 0.2850685060854935),
     ('G020021', 0.2848533988749895),
     ('G030260', 0.2831033988749895),
     ('G030347', 0.2803),
     ('G030350', 0.2803),
     ('G010113', 0.2786497224367903),
     ('G030269', 0.27765339887498947),
     ('G030020', 0.2774494448735806),
     ('G010108', 0.26891427437310833),
     ('G030080', 0.2675350269189626),
     ('G030027', 0.26740205761529995),
     ('G010006', 0.26655187852639733),
     ('G010273', 0.2654),
     ('G030119', 0.2654),
     ('G030241', 0.2654),
     ('G030016', 0.2654),
     ('G030047', 0.2654),
     ('G030129', 0.2654),
     ('G030203', 0.2654),
     ('G030233', 0.2654),
     ('G030040', 0.2654),
     ('G030211', 0.2654),
     ('G030214', 0.2654),
     ('G030043', 0.2654),
     ('G030263', 0.2625870929175277),
     ('G030227', 0.2601533988749895),
     ('G030025', 0.2583388279778489),
     ('G010238', 0.258059403398501),
     ('G030259', 0.2579125224324688),
     ('G010225', 0.2569),
     ('G010287', 0.2569),
     ('G010196', 0.2569),
     ('G010220', 0.2569),
     ('G010297', 0.2569),
     ('G010160', 0.2569),
     ('G010180', 0.2569),
     ('G010240', 0.2569),
     ('G010274', 0.2569),
     ('G010301', 0.2569),
     ('G010284', 0.2569),
     ('G010139', 0.2569),
     ('G010145', 0.2569),
     ('G010040', 0.2569),
     ('G010280', 0.2569),
     ('G030294', 0.2554435464587638),
     ('G020130', 0.25540862120548874),
     ('G020035', 0.25215679774997896),
     ('G030343', 0.2520577127364258),
     ('G030113', 0.24771220585942175),
     ('G030136', 0.24408777596066653),
     ('G030333', 0.2422),
     ('G030328', 0.2422),
     ('G030180', 0.2422),
     ('G030029', 0.2422),
     ('G030145', 0.2422),
     ('G010173', 0.24025339887498948),
     ('G010263', 0.24025339887498948),
     ('G010161', 0.24025339887498948),
     ('G020039', 0.24014114820126903),
     ('G020087', 0.23877494720807615),
     ('G030081', 0.2383965170602525),
     ('G020118', 0.2365354405893726),
     ('G030073', 0.23095181204301388),
     ('G010172', 0.2299824437144995),
     ('G020060', 0.22602170308492142),
     ('G020023', 0.2260033988749895),
     ('G020053', 0.22587252648670952),
     ('G010053', 0.22530187852639735),
     ('G030308', 0.22528041789032374),
     ('G030013', 0.22435039662398254),
     ('G010111', 0.2225),
     ('G010033', 0.2225),
     ('G010258', 0.2225),
     ('G010241', 0.2225),
     ('G010156', 0.2225),
     ('G010253', 0.2225),
     ('G010158', 0.2225),
     ('G010252', 0.22189972243679026),
     ('G010059', 0.21985632710696773),
     ('G020045', 0.21811434195681806),
     ('G030314', 0.21759121732666126),
     ('G030274', 0.21741652879244278),
     ('G030281', 0.21694354645876385),
     ('G030146', 0.21694354645876385),
     ('G010154', 0.21289972243679028),
     ('G010230', 0.2113033988749895),
     ('G030318', 0.20989574548966639),
     ('G010262', 0.2098),
     ('G010021', 0.2098),
     ('G010278', 0.2098),
     ('G010286', 0.2098),
     ('G010068', 0.2098),
     ('G010027', 0.2098),
     ('G010164', 0.2098),
     ('G030187', 0.20898891807222048),
     ('G010024', 0.20663299730846335),
     ('G030229', 0.20488109448379593),
     ('G010229', 0.199),
     ('G010151', 0.199),
     ('G010032', 0.199),
     ('G010058', 0.19745339887498947),
     ('G010182', 0.19745339887498947),
     ('G030150', 0.1964625224324688),
     ('G010037', 0.1964625224324688),
     ('G030076', 0.19399354645876385),
     ('G030130', 0.18845711732781847),
     ('G030152', 0.186925837490523),
     ('G020085', 0.18614812569270334),
     ('G010148', 0.18595339887498946),
     ('G030021', 0.1856177996249965),
     ('G020121', 0.1831533988749895),
     ('G010291', 0.1781533988749895),
     ('G020100', 0.1748),
     ('G020071', 0.1748),
     ('G020022', 0.1748),
     ('G020103', 0.1748),
     ('G020037', 0.1748),
     ('G020019', 0.1748),
     ('G020072', 0.1748),
     ('G020106', 0.1748),
     ('G020066', 0.1748),
     ('G020011', 0.1748),
     ('G020033', 0.1736033988749895),
     ('G030306', 0.1727320384512718),
     ('G010193', 0.1713),
     ('G010231', 0.1713),
     ('G010155', 0.1713),
     ('G010028', 0.1713),
     ('G010243', 0.1713),
     ('G010300', 0.1713),
     ('G010185', 0.1713),
     ('G010159', 0.1713),
     ('G010244', 0.1713),
     ('G010218', 0.1713),
     ('G010014', 0.1713),
     ('G010233', 0.1713),
     ('G010009', 0.1713),
     ('G010275', 0.1713),
     ('G010010', 0.1713),
     ('G010288', 0.16898333333333332),
     ('G010248', 0.16836405715288744),
     ('G020094', 0.1681458406409095),
     ('G020065', 0.16753422586484107),
     ('G010149', 0.16629818825631804),
     ('G030317', 0.16386785193335077),
     ('G010147', 0.16263502691896256),
     ('G010117', 0.15945447255899808),
     ('G030109', 0.15460339887498947),
     ('G010217', 0.15369500364742666),
     ('G020026', 0.1522542266341261),
     ('G030046', 0.1489033988749895),
     ('G030038', 0.1483),
     ('G010201', 0.1483),
     ('G010257', 0.1483),
     ('G010279', 0.1483),
     ('G010157', 0.1483),
     ('G010303', 0.1483),
     ('G010016', 0.1483),
     ('G010285', 0.1483),
     ('G010184', 0.14793314829119353),
     ('G020091', 0.14675339887498948),
     ('G030118', 0.14495339887498948),
     ('G020034', 0.1427),
     ('G020010', 0.1427),
     ('G020058', 0.1427),
     ('G020068', 0.1427),
     ('G030204', 0.1421033988749895),
     ('G030175', 0.141851421874285),
     ('G030177', 0.14050464407965088),
     ('G010199', 0.1402044725589981),
     ('G020014', 0.13954628212979198),
     ('G010255', 0.13444500364742668),
     ('G010174', 0.13307556509887897),
     ('G010242', 0.1327),
     ('G010001', 0.1327),
     ('G030064', 0.1327),
     ('G030161', 0.13218767702019604),
     ('G010134', 0.12512691321870775),
     ('G020018', 0.1236),
     ('G020002', 0.1236),
     ('G020077', 0.1236),
     ('G030162', 0.1218685441376969),
     ('G030320', 0.11713211732781847),
     ('G030028', 0.11180339887498948),
     ('G030104', 0.11180339887498948),
     ('G030246', 0.11180339887498948),
     ('G020004', 0.10722408974423879),
     ('G020082', 0.10635041642197512),
     ('G030006', 0.1049),
     ('G030319', 0.1049),
     ('G030213', 0.1049),
     ('G030268', 0.1049),
     ('G030231', 0.09985950627795187),
     ('G020127', 0.0989),
     ('G020075', 0.0989),
     ('G020097', 0.0989),
     ('G020051', 0.0989),
     ('G020083', 0.0989),
     ('G010237', 0.09711274175764675),
     ('G030087', 0.08681998608915836),
     ('G030171', 0.0856),
     ('G030070', 0.0856),
     ('G030279', 0.0856),
     ('G030341', 0.0856),
     ('G030235', 0.0856),
     ('G030221', 0.0856),
     ('G030355', 0.0856),
     ('G030194', 0.0856),
     ('G030045', 0.0856),
     ('G030083', 0.0856),
     ('G030285', 0.0856),
     ('G030283', 0.08370235074743107),
     ('G030191', 0.075517414490614),
     ('G030245', 0.0742),
     ('G030202', 0.0742),
     ('G030143', 0.0742),
     ('G030199', 0.0742),
     ('G030192', 0.0742),
     ('G030244', 0.0742),
     ('G030293', 0.0742),
     ('G030088', 0.0742),
     ('G030284', 0.0742),
     ('G030095', 0.07245533905932738),
     ('G030198', 0.07216878364870323),
     ('G020096', 0.0709224864308354),
     ('G020006', 0.0699),
     ('G020008', 0.0699),
     ('G020005', 0.0699),
     ('G020003', 0.0699),
     ('G020108', 0.0699),
     ('G020105', 0.0699),
     ('G020012', 0.0699),
     ('G030255', 0.06454972243679027),
     ('G020070', 0.06190442329069303),
     ('G030332', 0.0606),
     ('G030053', 0.05892556509887897),
     ('G020078', 0.0571),
     ('G020009', 0.0571),
     ('G020117', 0.0571),
     ('G020015', 0.0571),
     ('G020013', 0.0571),
     ('G020024', 0.0571),
     ('G020007', 0.0571),
     ('G020126', 0.0571),
     ('G020050', 0.0494),
     ('G020061', 0.0494),
     ('G030183', 0.0),
     ('G030151', 0.0),
     ('G030267', 0.0),
     ('G030168', 0.0),
     ('G030264', 0.0),
     ('G030126', 0.0),
     ('G030351', 0.0),
     ('G030048', 0.0),
     ('G030352', 0.0),
     ('G030049', 0.0),
     ('G030344', 0.0),
     ('G030261', 0.0),
     ('G030208', 0.0),
     ('G030215', 0.0),
     ('G030330', 0.0),
     ('G030154', 0.0),
     ('G030037', 0.0),
     ('G030305', 0.0),
     ('G030265', 0.0),
     ('G030023', 0.0),
     ('G030174', 0.0),
     ('G030349', 0.0),
     ('G030258', 0.0),
     ('G030063', 0.0),
     ('G030034', 0.0),
     ('G030290', 0.0),
     ('G030237', 0.0),
     ('G030005', 0.0),
     ('G030253', 0.0),
     ('G030304', 0.0),
     ('G030232', 0.0),
     ('G030289', 0.0),
     ('G030018', 0.0)]



## 5. 利用個人喜好分數，搭配三種推薦策略，建構推薦清單

### 5.1 推薦策略一：利用使用者收藏清單判斷其偏好戲劇類型，將這些類型的戲劇抽取出來，利用個人喜好分數進行排序


```python
def preferGenre(histroryDrama, user_ip):
    
    r_mix_dic=aveGrade()

    # 利用使用者過去的收藏清單判斷其偏好的戲劇類型(取前三種)
    df_drama_zone_genre_actor_ind=drama_feature_ind(histroryDrama, user_ip)
    drama_list_exclude=df_drama_zone_genre_actor_ind['dramaID'].tolist()

    index_list=list(df_drama_zone_genre_actor_ind.mean(axis=0).index)[3:28]
    grade_list=list(df_drama_zone_genre_actor_ind.mean(axis=0))[3:28]
    ind_preference={}
    for i in range(len(index_list)):
        ind_preference[index_list[i]]=grade_list[i]
    genre_prefer=sorted(ind_preference.items(), key=lambda d: d[1], reverse=True)
    
    #將這些類型的戲劇抽取出來
    df_Drama_prefer=df_Drama_GenreName[(df_Drama_GenreName['genreName']==genre_prefer[0][0]) \
                                       | (df_Drama_GenreName['genreName']==genre_prefer[1][0])\
                                       | (df_Drama_GenreName['genreName']==genre_prefer[2][0])]

    df_Drama_prefer_list=df_Drama_prefer['dramaID'].tolist()
                                       
    #利用個人喜好分數進行排序, 但排除使用者已經收藏的戲劇
    personal_rec={}
    for i in df_Drama_prefer_list:
        try:
            personal_rec[i]=r_mix_dic[i]        
        except:
            pass

    personal_rec1=sorted(personal_rec.items(), key=lambda x:-x[1])

    personal_rec1_list=[]
    for i in personal_rec1:
        if i[0] in drama_list_exclude:
            pass
        else:
            personal_rec1_list.append(i[0])

    return personal_rec1_list   
```

### 5.2 推薦策略二：利用豆辨評分找出前100部評分最高的戲劇，將這些戲劇抽取出來，利用個人喜好分數進行排序


```python
def popular(histroryDrama, user_ip):
    
    r_mix_dic=aveGrade()
    df_drama_zone_genre_actor_ind=drama_feature_ind(histroryDrama, user_ip)
    personal_rec1_list=preferGenre(histroryDrama, user_ip)
    drama_list_exclude=df_drama_zone_genre_actor_ind['dramaID'].tolist()

    #利用豆辨評分找出前100部評分最高的戲劇
    pop=tbl_Drama.sort_values('hitto', ascending=False).head(100)
    df_pop_prefer_list=pop['dramaID'].tolist()
    
    #利用個人喜好分數進行排序, 但排除使用者已經收藏或是已出現在「推薦策略一」的戲劇
    pop_rec={}
    for i in df_pop_prefer_list:
        try:
            pop_rec[i]=r_mix_dic[i]        
        except:
            pass

    pop_rec1=sorted(pop_rec.items(), key=lambda x:-x[1])

    pop_rec1_list=[]
    for i in pop_rec1:
        if i[0] in personal_rec1_list[0:3]+drama_list_exclude:
            pass
        else:
            pop_rec1_list.append(i[0])

    return pop_rec1_list
```

### 5.3 推薦策略三：找出近6個月出版的新劇，將這些戲劇抽取出來，利用個人喜好分數進行排序


```python
def latest(histroryDrama, user_ip):
    
    r_mix_dic=aveGrade()
    df_drama_zone_genre_actor_ind=drama_feature_ind(histroryDrama, user_ip)
    personal_rec1_list=preferGenre(histroryDrama, user_ip)
    pop_rec1_list=popular(histroryDrama, user_ip)
    drama_list_exclude=df_drama_zone_genre_actor_ind['dramaID'].tolist()
    
    #找出近6個月出版的新劇
    latest=tbl_Drama.sort_values('publishTime', ascending=False).head(60)
    df_latest_prefer_list=latest['dramaID'].tolist()
    
    #利用個人喜好分數進行排序, 但排除使用者已經收藏或是已出現在「推薦策略一」及「推薦策略二」的戲劇
    latest_rec={}
    for i in df_latest_prefer_list:
        try:
            latest_rec[i]=r_mix_dic[i]        
        except:
            pass

    latest_rec1=sorted(latest_rec.items(), key=lambda x:-x[1])
    
    latest_rec1_list=[]
    for i in latest_rec1:
        if i[0] in personal_rec1_list[0:3]+pop_rec1_list[0:3]+drama_list_exclude:
            pass
        else:
            latest_rec1_list.append(i[0])

    return latest_rec1_list
```

### 5.4 推薦清單


```python
histroryDrama=[('127.0.0.1', 'G020073'), ('127.0.0.1', 'G010012'), ('127.0.0.1', 'G010003'), ('127.0.0.1', 'G010002'), \
                  ('127.0.0.1', 'G010005'),('127.0.0.1', 'G010056'), ('127.0.0.1', 'G010169'), ('127.0.0.1', 'G020041') ]

personal_rec1_list=preferGenre(histroryDrama, '127.0.0.1')
pop_rec1_list=popular(histroryDrama, '127.0.0.1')
latest_rec1_list=latest(histroryDrama, '127.0.0.1')

print(personal_rec1_list[0:3])
print(pop_rec1_list[0:3])
print(latest_rec1_list[0:3])
```

    ['G030121', 'G010162', 'G030140']
    ['G010239', 'G010166', 'G010152']
    ['G010308', 'G010307', 'G010089']
    

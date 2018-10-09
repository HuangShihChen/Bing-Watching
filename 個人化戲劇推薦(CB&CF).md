
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
r_mix_dic_sorted[0:10]
```




    [('G030121', 0.6982310457736804),
     ('G010011', 0.692530265547432),
     ('G010162', 0.6882112859634918),
     ('G030140', 0.6862768429493377),
     ('G010136', 0.6852),
     ('G010025', 0.6675),
     ('G010109', 0.6675),
     ('G010103', 0.6675),
     ('G010264', 0.6634),
     ('G030092', 0.6592865978039979)]



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
    


# coding: utf-8

# ## Import the Librairies

# In[1]:

import pandas as pd 
import numpy as np
import boto
import pyprind
from scipy import stats
import scipy
import multiprocessing
import sys
import datetime
#%matplotlib inline


# In[2]:

N_threads = int(sys.argv[1])
print('N_threads')
print(N_threads)


# In[2]:

beg = int(sys.argv[2])
end = int(sys.argv[3])


# In[88]:

KNN_name = 'res_KNN_' + str(beg) +'_'+ str(end)


# ## Boto Stuff

# In[3]:

access = pd.read_csv('rootkey.csv', sep = '=', header = None)
s3 = boto.connect_s3(aws_access_key_id=access.loc[0,1], 
                     aws_secret_access_key=access.loc[1,1])


# In[4]:

s3_bucket_p2 = s3.get_bucket('rbuipracticals3')


# In[5]:

key_KNN = s3_bucket_p2.new_key(KNN_name + '.csv')


# ## Loading the Data

# In[8]:

#train = pd.read_csv('Data/train.csv')
#test = pd.read_csv('Data/test.csv')
#train0 = train


# In[7]:

train = pd.read_csv('https://s3.amazonaws.com/rbuipracticals3/train.csv')
test  = pd.read_csv('https://s3.amazonaws.com/rbuipracticals3/test.csv')


# In[8]:

#train = train0.loc[np.in1d(train0.artist, train0.artist.unique()[:20])] # Only use the 20 first artists
#train = train0


# In[9]:

#test = pd.read_csv('Data/test.csv')


# ## Description of the Algorithm

# 0. Compute the baseline
# 1. Compute the Common Support as : $NCommon_{ij}$ How many users rated/played both music
# 2. Compute the Rho_Correlation : $\rho(\bf{Y_{u_i}} - \bf{\overline{Y_u}} ; \bf{Y_{u_j}} - \bf{\overline{Y_u}})$
# 3. Compute the Similarity : $ \frac{N_Common \rho_{mj}}{N_Common + reg}$
# 4. Apply the Central Dogma : $Y_{um} = Y_{um}^{Baseline} + \frac{\sum_{j \in S^k(m)} s_{mj} (Y_{uj} - Y_{um}^{Baseline})}{\sum_{j \in S^k(m)} s_{mj}}$

# # I. Approach by filtering given u,m, don't compute all the support etc.

# ## Adding IDs ...

# In order to improve the speed of the various algorithm, we implement IDs instead of using the text to perform the location. 

# ##### Artist

# In[9]:

artist_id = pd.DataFrame(columns = ['artist'])
artist_id.loc[:, 'artist'] = train.artist.unique()
artist_id.loc[:, 'Id'] = artist_id.index
artist_id.index = artist_id.artist.values

train.loc[:, 'artist_id'] = artist_id.loc[train.artist, 'Id'].values
test.loc[:, 'artist_id'] = artist_id.loc[test.artist, 'Id'].values


# ##### User

# In[10]:

user_id = pd.DataFrame(columns = ['user'])
user_id.loc[:, 'user'] = train.user.unique()
user_id.loc[:, 'Id']   = user_id.index
user_id.index = user_id.user.values

train.loc[:, 'user_id'] = user_id.loc[train.user, 'Id'].values
test.loc[:, 'user_id']  = user_id.loc[test.user, 'Id'].values


# ## New Model Hybrid

# ### Description
# 1. Use the NSupport, and the Support precomputed
# 2. Create a matrix of the top Artist that will be used to scale down the basis of the artist used to compute the sim
# 3. Apply the classical algo

# ### Load the Relevant Data

# In[12]:

# user_avg = pd.read_csv('Data/EC2/user_avg.csv')
# support = np.load('Data/EC2/support.npy', encoding = 'bytes')
# N_support = np.load('Data/EC2/Nsupport.npy', encoding = 'bytes')


# In[ ]:

user_avg = pd.read_csv('user_avg.csv')
support = np.load('support.npy', encoding = 'bytes')
N_support = np.load('Nsupport.npy', encoding = 'bytes')


# In[ ]:

#user_avg = pd.read_csv('https://s3-us-west-1.amazonaws.com/practicals3/user_avg.csv')
#support = np.load('https://s3-us-west-1.amazonaws.com/practicals3/support.npy', encoding = 'bytes')
#N_support = np.load('https://s3-us-west-1.amazonaws.com/practicals3/Nsupport.npy', encoding = 'bytes')


# ### Baseline

# In[40]:

Y_bar = train.plays.median()
def computeBaseline2(u,m,user_avg,artist_avg,Y_bar):
    Y_u = user_avg.loc[u, 'AVG']
    #Y_m = artist_avg.loc[m, 'AVG']
    #Y_baseline = Y_bar + Y_u - Y_bar + Y_m - Y_bar
    Y_baseline = Y_u
    return Y_baseline


# ### Artist AVG

# In[14]:

prog_bar = pyprind.ProgBar(len(train.artist_id.unique()))
artist_avg = pd.DataFrame(columns=['Artist', 'AVG'])
for m in train.artist_id.unique():
    artist_avg = artist_avg.append({'Artist': m, 'AVG' : train.loc[train.artist_id == m].plays.median()}, ignore_index=True)
    prog_bar.update()


# ### Top Artist

# Let's rank the artists by their number of unique views

# In[15]:

artist_rank = train.artist_id.value_counts()
train.loc[:, 'Popularity_Index'] = train.artist_id.apply(lambda x : artist_rank[x])


# In[16]:

user_rank = train.user_id.value_counts()
#train.loc[:, 'UserPopularity_Index'] = train.user_id.apply(lambda x : user_rank[x])


# In[43]:

def computeSim(artist1, artist2, support, N_support, user_avg, user_rank, train, reg = 3, nb_common_user = 3):
    
    commonUser = support[artist1][artist2]
    
    # Only keep the 10 most popular user
    commonUser = user_rank[commonUser].sort_values(ascending = False).index[:nb_common_user]
    
    #N_Common = N_support[artist1][artist2]
    N_Common = len(commonUser)
    user_artist1 = []
    user_artist2 = []
    
    ix_artist1 = train.artist_id == artist1
    ix_artist2 = train.artist_id == artist2
    
    for i in commonUser:
        u_avg = user_avg.loc[i,'AVG']
        user_artist1.append(int(train.loc[(train.user_id == i) & ix_artist1].plays) - u_avg)
        user_artist2.append(int(train.loc[(train.user_id == i) & ix_artist2].plays) - u_avg)
    rho = scipy.stats.pearsonr(user_artist1, user_artist2)[0]
    rho_shrunk = N_Common * rho / (N_Common + reg) 
    return((1-rho_shrunk)/2)


# ##### Apply the Basic Algo

# In[101]:

def MakePrediction(user, artist, base_artist):#, support, N_support, user_avg, user_rank, artist_avg, Y_bar, train):

    u = user 
    artist1 = artist
    
    # Compute the sim
    sim = np.empty(shape = (len(base_artist)))
    #prog_bar = pyprind.ProgBar(len(base_artist))

    #train_small
    # ix = np.in1d(train.artist_id, np.append(m, base_artist))
    # train_small = train.loc[ix]

    for j,artist2 in enumerate(base_artist):
        sim[j] = computeSim(artist1, artist2, support, N_support, user_avg, user_rank, train)
        #prog_bar.update()

    #input:
    k = 3

    res = pd.DataFrame(columns = ['artist_id', 'sim'])
    res.loc[:, 'artist_id'] = base_artist
    res.loc[:, 'sim'] = sim
    res = res.sort_values(by = 'sim', ascending = False)

    # Apply the final Algo
    num = 0
    denom = 0
    Yum_base = computeBaseline2(u,m,user_avg, artist_avg, Y_bar)

    for i in res.index[:k]:
        Yuj = int(train[(train.user_id == u) & (train.artist_id == res.loc[i, 'artist_id'])].plays)
        num += res.loc[i, 'sim']*(Yuj-Yum_base)
        denom += res.loc[i, 'sim']

    Y_um = Yum_base + num/denom

    return Y_um


# In[102]:

def MakePrediction2(r):
    base_artist = train.loc[train.user_id == r[1]['user_id'], ['artist_id', 'Popularity_Index']].sort_values(by = 'Popularity_Index', ascending = False).artist_id[:5]
    temp = MakePrediction(r[1]['user_id'], r[1]['artist_id'], base_artist)#, support, N_support, user_avg, user_rank, artist_avg, y_bar, train)
    return temp


# ### Run in Threads

# In[22]:


pool = multiprocessing.Pool(N_threads)


# In[103]:

#%%time
print('start 50')
print (datetime.datetime.today())
res = pool.map(MakePrediction2, test.head(50).iterrows())
print (datetime.datetime.today())
print('end 50')


# In[82]:

print('start')
print (datetime.datetime.today())
final_res = pd.DataFrame(columns = ['Id', 'plays'])
final_res.loc[:, 'Id'] = test[beg:end].Id.values
res = pool.map(MakePrediction2, test[beg:end].iterrows())
final_res.loc[:, 'plays'] = res
print (datetime.datetime.today())
print('end')


# ## Run With Apply

# In[71]:

#test.head(50).apply(lambda x : MakePrediction2((0, x)), axis = 1)


# In[78]:

#cProfile.run('test.head(10).apply(lambda x : MakePrediction2((0, x)), axis = 1)', )


# In[79]:

# %%time
# res = test.head(50).apply(lambda x : MakePrediction2((0, x)), axis = 1)


# ## Save

# In[ ]:

final_res.to_csv(KNN_name + '.csv', index = False)


# In[ ]:

key_KNN.set_contents_from_filename(KNN_name + '.csv')


# In[ ]:




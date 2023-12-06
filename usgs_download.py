# %%
import pandas as pd
import os
import urllib.request
from urllib.request import HTTPError




    
url = 'https://waterservices.usgs.gov/nwis/iv/?sites=06620000&parameterCd=00060&startDT=2017-05-28T15:50:46.705-07:00&endDT=2023-12-05T15:50:46.705-07:00&siteStatus=all&format=rdb'


fetched_request = urllib.request.urlopen(url)

with open(os.getcwd() + os.sep + 'loc1cfs', 'wb') as f:
    f.write(fetched_request.read())


# %%
loc1 = pd.read_csv('loc1',sep='\t',engine='python',skiprows=28)

loc2 = pd.read_csv('loc2',sep='\t',engine='python',skiprows=28)
#%%

loc1cfs = pd.read_csv('loc1cfs',sep='\t',engine='python',skiprows=30)

loc2cfs = pd.read_csv('loc2cfs',sep='\t',engine='python',skiprows=30)

#%%
times = loc1.loc[(loc1['14n']>8)&(loc1['14n']<8.03)]['20d'].values
loc1cfs.loc[loc1cfs['20d'].isin(times)]['14n'].hist()
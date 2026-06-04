#!/usr/bin/env python
# coding: utf-8


import glob
from joblib import load
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import numpy as  np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
import datetime
from matplotlib.dates import MonthLocator, DateFormatter
import datetime as dt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FuncFormatter
# from matplotlib.dates import YearLocatorcator
from scipy.stats import pearsonr as p


'''
sal= xr.open_dataset('COP_salt_TA.nc').so
sal

temp= xr.open_dataset('COP_sst_TA.nc').thetao
temp


mld= xr.open_dataset('COP_mld_TA.nc').mlotst
mld['time'] = np.array(temp.time)
mld

temp_new1 = temp
sal_new1 = sal


print(temp_new1.shape,mld.shape,sal_new1.shape)


temp_reshaped1=np.array(temp_new1).reshape(336*313*600)
sal_reshaped1=np.array(sal_new1).reshape(336*313*600)
mld_reshaped1=np.array(mld).reshape(336*313*600)


id2 = np.append(np.where(np.isnan(sal_reshaped1)),np.where(np.isnan(mld_reshaped1)))

id3 = np.append(id2,np.where(np.isnan(temp_reshaped1)))

temp_reshaped1[id3] = np.nan
sal_reshaped1[id3] = np.nan
mld_reshaped1[id3] = np.nan

file2 = pd.DataFrame()

file2['SST'] = temp_reshaped1
file2['SSS'] = sal_reshaped1
file2['MLD'] = mld_reshaped1

file2['MLD'] = np.log(file2['MLD'])


file2.head()


# In[9]:


file2.isnull().sum()


# In[10]:


X = pd.read_csv("X_train_42.csv")
X = X.drop(['Date','Longitude [E]','Latitude [N]'],axis=1)
X.head()


# In[11]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
file3 = sc.transform(file2)



a = pd.DataFrame(file3)
a.head()


files = glob.glob(f'model/*')
# In[ ]:

print('.....................starting.........................')
for i in np.arange(0,140,1):
    xgb=load(files[i]) 
    # xgb = load(f'/home/NCMRWF/fvcom/Apurva/TA/model/xgb_BoB.joblib_TA_{i+1}.dat')
    pred2 = xgb.predict(file3)
    print('prediction done')
    pred2[id3] = np.nan
    pred2 = pred2.reshape(336,313,600)

    da12 = xr.DataArray(

        data=pred2,

        dims=["Time","Lat","Lon"],name="TA",

        coords={'Time':temp.time.to_numpy(),'Lat':temp.latitude.to_numpy(),'Lon':temp.longitude.to_numpy()},

        attrs=dict(

            description="TA",

            units="micromol/kg",

        ),

    )
   # da13 = da12.groupby('Time.month').mean()
   # da13['month'] = pd.date_range(start="01-01-2007",periods=12,freq="1M")
   # da13 = da13.rename({"month":"Time"})
    print('done')
    da12.to_netcdf(f"Pred_latest/Pred_inter_TA_{i+1}.nc")
    print(f":::predicted TA for model xgb_BoB.joblib_TA_{i+1}.dat::::")

'''
flist_pred=glob.glob('Pred_latest/Pred_inter_TA*.nc')

mod=xr.open_mfdataset(flist_pred,concat_dim='model',combine='nested')

#mean=mod.mean(dim='model')
std=mod.std(dim='model')

#mean.to_netcdf('TA_final_ensemble.nc')
std.to_netcdf('TA_final_uncertainty.nc')


# %%
import glob
import rioxarray as rxr
import numpy as np

filenames_1 = glob.glob('sar//flood//'+'*.tif')
filenames_2 = glob.glob('sar//noflood//'+'*.tif')


for image in filenames_1:
    t = rxr.open_rasterio(image).sel(band=1).values
    np.save(image.split('.tif')[0]+'.npy', t)

# %%
for image in filenames_2:
    t = rxr.open_rasterio(image).sel(band=1).values
    np.save(image.split('.tif')[0]+'.npy', t)
# %%

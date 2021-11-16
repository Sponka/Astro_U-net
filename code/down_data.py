from astroquery.mast import Catalogs
from astroquery.vizier import Vizier
from astroquery.mast import Observations
import urllib.request
import os


filter_wide = ["F555W","F606W", "F814W"]
for f in filter_wide:
    if not os.path.isdir("data/" + f + "/"):
        os.makedirs("data/" + f + "/")
    obsTable = Observations.query_criteria(calib_level= 3, dataproduct_type = 'image',
                                               obs_collection = ["HLA"],
                                        instrument_name = ["WFC3/UVIS"], filters = [f])

    for url in obsTable['dataURL']:
        if isinstance(url, type(obsTable[7]['dataURL'])):  #to have valid url, example: https://hla.stsci.edu/cgi-bin/getdata.cgi?dataset=hst_11426_21_wfc3_uvis_f606w_drz.fits
            if url[-5:] == str(".fits"):
                file = "data/" + f + "/" + str(url[50:])
                urllib.request.urlretrieve(url,file)
 
#This code will download all fits images with filters F555W and F606W. 
#Example of valid url https://hla.stsci.edu/cgi-bin/getdata.cgi?dataset=hst_11426_21_wfc3_uvis_f606w_drz.fits 
#To filter used data the dataset.txt can be used
#If image appears more times in dataset.txt it means that image was cut into multiple images


import os
import glob
import pandas as pd
import time
from PIL import Image
dir = "/home/lwang/data/facenetsmall/lfwvalid"
files = glob.glob(dir+"/*/*")

verbosity     = 1
which_dataset = 1


time0 = time.time()
df = pd.DataFrame()
print("The number of files: ", len(files))
for idx, file in enumerate(files):
    if idx%10000 == 0:
        print("[{}/{}]".format(idx, len(files)-1))
    '''    
    try:
        img = Image.open(file) # open the image file
        img.verify() # verify that it is, in fact, an image
    except (IOError, SyntaxError) as e:
        if verbosity == 1:
            print('Bad file:', file) # print out the names of corrupt files
        pass
    else:
        face_id    = os.path.basename(file).split('.')[0]
        face_label = os.path.basename(os.path.dirname(file))
        df = df.append({'id': face_id, 'name': face_label}, ignore_index = True)
    '''    
    face_id    = os.path.basename(file).split('.')[0]
    face_label = os.path.basename(os.path.dirname(file))
    df = df.append({'id': face_id, 'name': face_label}, ignore_index = True)


df = df.sort_values(by = ['name', 'id']).reset_index(drop = True)


df['class'] = pd.factorize(df['name'])[0]

dname = os.path.dirname(dir)
fname = os.path.basename(dir) + ".csv"
fw = os.path.join(dname, fname)

df.to_csv(fw, index = False)


elapsed_time = time.time() - time0
print("elapsted time: ", elapsed_time//3600, "h", elapsed_time%3600//60, "m")






import pandas as pd
import sqlite3
from collections import OrderedDict
from simple_object_tracking.pyimagesearch.centroidtracker import CentroidTracker



conn = sqlite3.connect('/Users/pascal/Coding/MP_bees/bees.db')
c = conn.cursor()

df = pd.read_sql_query("select * from coordinates", conn)

df_track = pd.DataFrame()
df_track['bee_id'] = 0

# first frame
objects = OrderedDict()
frame=1
run = 1
image = df[(df.frame==frame)&(df.run_id==run)]
for i,row in image.iterrows():
    bee = {i:[(row.x_center,row.y_center)]}
    objects.update(bee)

frame = 2
run = 1
image = df[(df.frame==frame)&(df.run_id==run)]
for i,row in image.iterrows():

    bee = {i:[(row.x_center,row.y_center)]}
    print(bee)
    # objects.update(bee)

first_bee = []
frames = df.frame.unique()
run = 1
ct = CentroidTracker()
for frame in range(1,len(frames)+1):
    image = df[(df.frame == frame) & (df.run_id == run)]
    rects = []
    for i,row in image.iterrows():
        rects.append([row.x_start,row.x_end,row.y_start,row.y_end])
    objects = ct.update(rects)
    print(frame)
    first_bee.append(objects[0])

import numpy as np
import math

def angle_between(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return np.rad2deg((ang1 - ang2) % (2 * np.pi))

for i in range(len(first_bee)):
    p1 = (first_bee[i][0],first_bee[i][1])
    p2 = (first_bee[i+1][0],first_bee[i+1][1])
    distance = math.hypot(first_bee[i+1][0]-first_bee[i][0],first_bee[i+1][1]-first_bee[i][1])
    print(angle_between(p1,p2),distance)

import pandas as pd
import xml.etree.ElementTree as ET
import os
import sqlite3

XML_PATH = "/Users/pascal/Coding/MP_bees/images/test"
xml_lst = []
for filename in os.listdir(XML_PATH):
    if not filename.endswith('.xml'): continue
    fullname = os.path.join(XML_PATH, filename)
    xml_lst.append(fullname)

dfcols = ['filename', 'width', 'height', 'xmin','xmax','ymin','ymax','x_center','y_center']
df = pd.DataFrame(columns=dfcols)


for xml in xml_lst:
    etree = ET.parse(xml)
    root = etree.getroot()

    filename = root.find('filename').text
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    objects = root.findall('object')
    for object in objects:
        box = object.find('bndbox')
        xmin = int(box.find('xmin').text)
        ymin = int(box.find('ymin').text)
        xmax = int(box.find('xmax').text)
        ymax = int(box.find('ymax').text)
        x = (xmin + xmax) // 2
        y = (ymin + ymax) // 2
        df = df.append(
            pd.Series([filename,width,height,xmin,xmax,ymin,ymax,x,y],index = dfcols), ignore_index=True)

df.head()

conn = sqlite3.connect('/Users/pascal/Coding/MP_bees/simple_object_tracking/bees.db')
c = conn.cursor()

# c.execute("drop table ground_truth")
# c.execute("""create table ground_truth (
#             filename text,
#             width integer,
#             height integer,
#             xmin integer,
#             xmax integer,
#             ymin integer,
#             ymax integer,
#             x_center integer,
#             y_center integer
# )""")
df.to_sql("ground_truth",conn,if_exists='replace',index=False)


from json import loads as strTolist
import requests
li = '(-1,)'
li=li.replace('(',',').replace(')',',')
print(li)
data = li.split(',')
data_list = []
for d in data:
    if len(d) != 0:
        data_list.append(int(d))
data_shape = tuple(data_list)
print(data_shape)

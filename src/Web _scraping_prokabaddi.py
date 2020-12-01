#!/usr/bin/env python
# coding: utf-8

# In[3]:


from bs4 import BeautifulSoup
import json
import csv
import pandas as pd
import os
import selenium
from selenium import webdriver
import time
from PIL import Image
import io
import requests


# In[55]:


driver = webdriver.Firefox(executable_path = "/home/agrim/Downloads/firefox_webdriver/geckodriver")


# In[7]:


driver.get('https://www.prokabaddi.com/matchcentre/1687-scorecard')
n = driver.find_elements_by_class_name('teamA-score')
while not n:
    n = driver.find_elements_by_class_name('teamA-score')


# In[8]:


n = driver.find_elements_by_class_name('teamA-score')


# In[9]:


n


# In[5]:


link = []
for i in range(1, 61):
    st = f'{i}'
    link.append("https://www.prokabaddi.com/matchcentre/"+st+"-scorecard")
for i in range(62, 242):
    st = f'{i}'
    link.append("https://www.prokabaddi.com/matchcentre/"+st+"-scorecard")
for i in range(286, 424):
    st = f'{i}'
    link.append("https://www.prokabaddi.com/matchcentre/"+st+"-scorecard")
for i in range(625,763):
    st = f'{i}'
    link.append("https://www.prokabaddi.com/matchcentre/"+st+"-scorecard")
for i in range(1685, 1822):
    st = f'{i}'
    link.append("https://www.prokabaddi.com/matchcentre/"+st+"-scorecard")


# In[6]:


link.index("https://www.prokabaddi.com/matchcentre/404-scorecard")


# In[7]:


link.remove("https://www.prokabaddi.com/matchcentre/404-scorecard")


# In[56]:


team = []
score = []
score_op = []
count = 0
for l in link:
#     count += 1
#     if count < 635:
#         continue
    driver.get(l)
    nav = driver.find_elements_by_class_name('sipk-tabName.si-tab')
    while not nav:
        nav = driver.find_elements_by_class_name('sipk-tabName.si-tab')
    nav[2].click()

    d = driver.find_elements_by_class_name('sipk-headTohead-block')
    while not d:
        d = driver.find_elements_by_class_name('sipk-headTohead-block')
    a = driver.find_elements_by_class_name('teamA-score')
    b = driver.find_elements_by_class_name('teamB-score')
    team.append(d[0].text)
    print(count)
    if a:
        score.append(a[0].text)
    if b:
        score_op.append(b[0].text)
#     if a and b:    
#         if int(a[0].text) > int(b[0].text):
#             win.append('0')
#         else:
#             win.append('1')
#     c = driver.find_elements_by_class_name('si-fullName')
#     print(c)
#     if c:
#         team.append(c[0].text + " " + c[1].text)
#         op_team.append(c[2].text + " " + c[3].text)
#     d = driver.find_elements_by_class_name('sipk-hth-progressBlock')
#     print(d)
#     raid.append(n[0].text.split('\n')[1])
#     raid_op.append(n[0].text.split('\n')[2])
#     tackle.append(n[1].text.split('\n')[1])
#     tackle_op.append(n[1].text.split('\n')[2])
#     allout.append(n[2].text.split('\n')[1])
#     allout_op.append(n[2].text.split('\n')[2])
#     extra.append(n[3].text.split('\n')[1])
#     extra_op.append(n[3].text.split('\n')[2])
    


# In[211]:


driver.get("https://www.prokabaddi.com/matchcentre/1688-scorecard")
# d = driver.find_elements_by_class_name('sipk-headTohead-block')
# while not d:
#     d = driver.find_elements_by_class_name('sipk-headTohead-block')
nav = driver.find_elements_by_class_name('sipk-tabName.si-tab')
while not nav:
    nav = driver.find_elements_by_class_name('sipk-tabName.si-tab')
nav[2].click()

d = driver.find_elements_by_class_name('sipk-headTohead-block')


# In[230]:


import csv


# In[60]:


vals = []
for i in range(1,653):
    vals.append(team[i-1].split('\n'))


# In[62]:


len(vals)


# In[69]:


file = open('data_MLfinal.csv', 'w', newline ='') 


# In[ ]:


sc = score
op = score_op


# In[65]:


win = []
for i in range(1,653):
    if sc[i-1] > op[i-1]:    
        win.append('0')
    else:
        if sc[i-1] == op[i-1]:
            win.append('draw')
        else:
            win.append('1')


# In[66]:


win


# In[68]:


season = []
for i in range(1,653):
    if i < 61: 
        season.append('season 1')
    else:
        if i > 60 and i < 121:
            season.append('season 2')
        else:
            if i > 120 and i < 181:
                season.append('season 3')
            else:
                if i > 180 and i < 241:
                    season.append('season 4')
                else:
                    if i > 240 and i < 378:
                        season.append('season 5')
                    else:
                        if i > 377 and i < 515:
                            season.append('season 6')
                        else:
                            season.append('season 7')


# In[70]:


with file: 
    # identifying header   
    header = ['Team','Op Team','score','Opscore','RaidPoints','OpRaid','TacklePoints','OpTackle','Allout','OpAllOut','ExtraPoints','OpExtra','win','season'] 
    writer = csv.DictWriter(file, fieldnames = header) 
    writer.writeheader()
    for i in range(1,653):
        writer.writerow({'Team' : vals[i-1][0],
                         'Op Team': vals[i-1][2],
                         'score': sc[i-1],
                         'Opscore': op[i-1],
                         'RaidPoints': vals[i-1][4],
                         'OpRaid': vals[i-1][5],
                         'TacklePoints':vals[i-1][7],
                         'OpTackle': vals[i-1][8],
                         'Allout': vals[i-1][10],
                         'OpAllOut': vals[i-1][11],
                         'ExtraPoints': vals[i-1][13],
                         'OpExtra': vals[i-1][14],
                         'win': win[i-1],
                        'season': season[i-1]}
        )


# In[71]:


df = pd.read_csv("data_MLfinal.csv")


# In[72]:


df


# In[ ]:





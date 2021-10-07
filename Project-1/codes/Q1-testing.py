#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import collections
import matplotlib.pyplot as plt
# get_ipython().system('ls')


# In[2]:


df = pd.read_csv("../football_data.csv", index_col="Unnamed: 0")
# df = df.astype(str)
df


# In[3]:


# info for all dataset columns_name, dataType, null_count
df.info()


# In[4]:


df.shape


# In[5]:


# describe of data min, max, mean, std values for all columns
df.describe(include='all')


# In[43]:


clubs = set(df["Club"].tolist())
dict = {}
dict_new = {}

for i in clubs:
    dict[i] = []
    
for i in range(len(df.axes[0])):
    zval = df["Wage"].tolist()[i]
    if '€' in zval: zval = zval[1:]
    if 'K' in zval: zval = float(zval[:-1])*1000
    elif 'M' in zval:
        zval = zval[:-1]
        zval = float(zval)*1000000
    dict[df["Club"].tolist()[i]].append(float(zval))
    
for i in clubs:
    dict_new[i] = np.average(np.array(dict[i]))

dict_new = sorted(dict_new.items(), key = lambda kv:(kv[1], kv[0]), reverse=True) 


# In[58]:


k = collections.Counter(dict_new)
high = k.most_common(10)

club_name = []
club_wage = []
 
for i in high:
    club_name.append(i[0][0])
    club_wage.append(i[0][1])
    
plt.figure(figsize=(15,7))
plt.bar(club_name,club_wage)
plt.xticks(rotation=70)
plt.title("Clubs with highest wages", color="black")
plt.show()


# In[66]:


clubs = set(df["Nationality"].tolist())
dict = {}
dict_new = {}

for i in clubs:
    dict[i] = []
    
for i in range(len(df.axes[0])):
    zval = df["Overall"].tolist()[i]
    dict[df["Nationality"].tolist()[i]].append(float(zval))
    
for i in clubs:
    dict_new[i] = len(dict[i])

dict_new = sorted(dict_new.items(), key = lambda kv:(kv[1], kv[0]), reverse=True) 


# In[67]:


k = collections.Counter(dict_new)
high = k.most_common(10)

club_name = []
club_wage = []
 
for i in high:
    club_name.append(i[0][0])
    club_wage.append(i[0][1])
    
plt.figure(figsize=(15,7))
plt.bar(club_name,club_wage)
plt.xticks(rotation=70)
plt.title("Number of players from each country", color="black")
plt.show()


# In[8]:


# Wage count in ranges
# ShotPower vs age, Position (SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower', 'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression')
# Wage vs Value


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

X = df['Height'].tolist()

new = []

for i in X:
    if "'" in str(i):
        hh = i.split("'")
        hhh = int(hh[0])*12 + int(hh[1])
        new.append(hhh)
        

# plt.hist(new,bins=[65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81])
# plt.xlabel("height")
# plt.ylabel('freq')
# plt.show()


Y = df['Weight'].tolist()
# print(Y)

newY = []

for i in Y:
    if "lbs" in str(i):
        hh = i.split("l")
        hhh = int(hh[0])
        newY.append(hhh)
        
# plt.scatter(new,newY)
# plt.xlabel('Height')
# plt.ylabel('Weight')
# plt.show()



# # group data by Nationlity and sort it by number of player to get most country have player.
# national_player = df[['Nationality',"ID"]].groupby(by=['Nationality'],as_index=False).count().sort_values("ID",ascending=False)
# national_player.rename(columns = {'Nationality':"country", 'ID':'player_count'}, inplace = True)
# national_player = national_player.reset_index()
# national_player = national_player.drop(["index"], axis=1)
# national_player.head(10)
# print(national_player.head(10))



# # Slicing first 10 row form country player_count dataset
# player_count = national_player.iloc[0:10,1]
# national = national_player.iloc[0:10,0]

# # select seaborn style of chart to make display more good for eyes.
# plt.style.use("seaborn")
# # create bar chart between most 10 country and no. of player 
# plt.bar(national,player_count)
# plt.xticks(rotation=45)
# plt.title('Top 10 Country that have player in FIFA 19')
# plt.show()

############################ Everthing above this works, just uncomment

import scipy as sp

# slicing Age column and group it and count no. of player have same age for all age.
player_age = df[['Age',"ID"]].groupby(by=['Age'],as_index=False).count().sort_values("ID",ascending=False)
player_age.rename(columns = {'ID':'count'}, inplace = True)
player_age = player_age.reset_index().drop(["index"], axis=1)
player_age.head()



# display histogram of age for all player and fit normal distribution line for it.
_,bins,_ = plt.hist(df.Age,bins=df.Age.max()-df.Age.min(),label="Age with no. of player")
mu, sigma = sp.stats.norm.fit(df.Age)
best_fit_line = sp.stats.norm.pdf(bins, mu, sigma)
plt.plot(bins, df.shape[0] * best_fit_line,label="fit_line",color="red")

plt.title('Distrbution of Age with players in FIFA 19')
plt.ylabel("no. of player")
plt.xlabel("Age of player")
plt.legend()
plt.show()

preferred_foot = df.groupby("Preferred Foot")["Preferred Foot"].count()
preferred_foot
# plot pie chart to display the percentage for every foot that players preferred
plt.pie(preferred_foot, labels=["left","right"], explode=[0.1,0], autopct='%1.2f%%',colors=["#ea157a","#0089af"])
plt.legend()
plt.show()




# count number for every position in playground that have players and sorted it.
player_position = df[["Position","ID"]].groupby(by=['Position'],as_index=False).count().sort_values("ID",ascending=False)
player_position.rename(columns = {'ID':'count'}, inplace = True)
player_position = player_position.reset_index().drop(["index"], axis=1)
player_position.head()


# plot bar chart to display the number of players for every position with sorted.
plt.figure(figsize=(15,7))
plt.bar(player_position["Position"],player_position["count"])
plt.xticks(rotation=70)
plt.title("Player's Position Distrbution", color="black")
plt.show()

# get top 10 ST players in the world.
ST_position = df[df["Position"]=="ST"].sort_values("Overall",ascending=False)[["Name","Overall"]]
ST_position = ST_position.iloc[:10,:]

RF_position = df[df["Position"]=="RF"].sort_values("Overall",ascending=False)[["Name","Overall"]]
RF_position = RF_position.iloc[:10,:]

# function plot bar chart for top 10 player in selected position.
def draw(df, color, position, ax):
    plt.style.use('tableau-colorblind10')
    sns.barplot(df["Name"],df["Overall"],color=color , ax=ax).set_title("Most Top 10 " + position +" players", fontsize=14)
    ax.set_xticklabels(ax.get_xticklabels(),rotation=40)

# plot 4 figures that display Top 10 player in ST, GK, LW, RF positions.
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=[20, 5])

draw(ST_position,"#0089af", "ST",axes[1])
draw(RF_position,"#72bd35", "RF",axes[0])
plt.show()


# In[9]:


# plot the distribution of overall rating.
plt.figure(figsize=(15, 7))
sns.countplot(df.Overall, label="overall_hist",color="#c81067")
plt.title("Overall rating distribution for all Player")
plt.legend()
plt.show()


# In[12]:


import plotly.graph_objects as go

attribute_dict = {"shooting" : ["Positioning","Finishing","ShotPower","LongShots","Volleys","Penalties"],
                  "passing" : ["Vision","Crossing","FKAccuracy","ShortPassing","LongPassing","Curve"],
                  "dribbling" : ["Agility","Balance", "Reactions", "BallControl", "Dribbling","Composure"],
                  "defending" : ["Interceptions", "HeadingAccuracy", "Marking", "StandingTackle","SlidingTackle"],
                  "physical" : ["Jumping", "Stamina", "Strength","Aggression"]}

# funcion that calcualte attribute for any player: need using player index
def calculate_attribute(dataframe,player_index):
    allcols = []
    
    for i in attribute_dict.values():
        allcols.extend(i) 
        
    player_observation = dataframe.loc[player_index,allcols].astype("int64")
    player_skills = []
    
    for i in attribute_dict.keys():
        lis = attribute_dict.get(i)
        player_skills.append(int(sum(player_observation[lis])/len(player_observation[lis])))

    return {i.upper()+": "+str(j)+"%":j for i,j in zip(attribute_dict.keys(),player_skills)}


# function plot radar diagram for any player, need player skills and player name.
def plot_player_radar(skills,player_name):
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(r=list(skills.values()),theta=list(skills.keys()),fill='toself',
                                  name=player_name, line_color='darkviolet',))

    fig.update_layout(polar=dict(radialaxis=dict(visible=False,range=[0,100])),showlegend=True)
    fig.show()

# draw attribute Details radar chart for RONALDO.

player_index = 0
player_skills = calculate_attribute(df,player_index)
plot_player_radar(player_skills,df.iloc[player_index]["Name"])

player_index = 1
player_skills = calculate_attribute(df,player_index)
plot_player_radar(player_skills,df.iloc[player_index]["Name"])

player_index = 2
player_skills = calculate_attribute(df,player_index)
plot_player_radar(player_skills,df.iloc[player_index]["Name"])


# In[13]:


# player position in each line in playground.
position = {"deffender":["RB","LB","CB","LCB","RCB","RWB","RDM","CDM","LDM","LWB"],
             "midder":["RM","LM","CM","LCM","RCM","RAM","CAM","LAM"],
             "attacker":["RW","RF","CF","LF","LW","RS","ST","LS"]
            }
lineup = [3,4,3]

# function that get best squad in the world based on your Lineup.
def get_best_squad(lineup):
    best_squad = df[df.Position == "GK"].sort_values("Overall",ascending=False).iloc[0:1]
    for j, k in zip(position.keys(),range(3)):
        best = []
        for i in position.get(j):
            best.append(df[df.Position == i].sort_values(["Overall","Potential"], ascending=[False,False]).iloc[0])
        best = pd.DataFrame(best).sort_values(["Overall","Potential"], ascending=[False,False])
        best = best.iloc[0:lineup[k]]
        best_squad = pd.concat([best_squad, best])
    return best_squad

# get best squad on the world based on lineup which you select.
best_sqaud = get_best_squad(lineup)
best_sqaud.reset_index(inplace=True)
player_index = list(best_sqaud.loc[:,["index"]].values.reshape(11,))
best_sqaud.drop("index",axis=1,inplace=True)
best_sqaud


# In[14]:



X = df['Value'].tolist()
Y = df['Overall'].tolist()
Z = df['Wage'].tolist()

Xnew = []
Ynew = []
Znew = []

for i in range(len(X)):
    
    xval = X[i]
    yval = Y[i]
    zval = Z[i]
    
    if '€' in xval: xval = xval[1:]
    if 'K' in xval: xval = float(xval[:-1])*1000
    elif 'M' in xval:
        xval = xval[:-1]
        xval = float(xval)*1000000

    
    if '€' in zval: zval = zval[1:]
    if 'K' in zval: zval = float(zval[:-1])*1000
    elif 'M' in zval:
        zval = zval[:-1]
        zval = float(zval)*1000000
        
    xval=float(xval)
    yval=float(yval)
    zval=float(zval)
    
    if xval!=0.0 and yval!=0.0 and zval!=0.0:
        Xnew.append(float(xval))
        Ynew.append(float(yval))
        Znew.append(float(zval))
    


plt.scatter(Ynew,Xnew)
plt.xlabel('Overall')
plt.ylabel('Value')
plt.show()

plt.scatter(Ynew,Znew)
plt.xlabel('Overall')
plt.ylabel('Wage')
plt.show()

plt.scatter(Xnew,Znew)
plt.xlabel('Value')
plt.ylabel('Wage')
plt.show()


# In[72]:


# Wage count in ranges
# ShotPower vs age, Position (SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower', 'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression')
# Wage vs Value


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('football_data.csv')

toKeep=np.array([3,7,8,11,12,13,15,16,17,22,23])
for i in range(25,89):
    toKeep=np.append(toKeep,i)

X = dataset.iloc[:,toKeep].values
header=dataset.iloc[:,toKeep].columns

print(len(X))
print(X.shape[1]-1)

print(dataset.info())


for j in np.array([3,4,X.shape[1]-1]):
    for i in range(X.shape[0]):
        if type(X[i,j]) is float:
            pass
        elif X[i,j][-1]=='M':
            X[i,j]=float(X[i,j][1:-1])*1000000
        elif X[i,j][-1]=='K':
            X[i,j]=float(X[i,j][1:-1])*1000
        else:
            X[i,j]=0


plt.scatter(X[:,1],X[:,4])
plt.xlabel('Overall')
plt.ylabel('Wage')
plt.show()

plt.scatter(X[:,1],X[:,3])
plt.xlabel('Overall')
plt.ylabel('Value')
plt.show()

plt.scatter(X[:,1],X[:,74])
plt.xlabel('Overall')
plt.ylabel('Release Clause')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[22]:


height = collections.Counter(df['Height'].values.tolist())


# In[24]:


plt.bar(list(height.keys()), list(height.values()))
xlocs=[i+1 for i in range(len(list(height.keys())))]
# for i, v in enumerate(list(height.values())):
#     plt.text(xlocs[i] - 1.75, v + 0.02, str(v))
plt.show()


# In[ ]:





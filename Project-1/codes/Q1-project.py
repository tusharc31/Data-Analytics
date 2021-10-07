#imports
import numpy as np
import pandas as pd
import plotly.graph_objects
import matplotlib.pyplot
import seaborn
import collections
import scipy

print("\n\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n\n")
df = pd.read_csv("../football_data.csv", index_col="Unnamed: 0")
print("Data set after extracting from CSV:\n")
print(df)

print("\n\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n\n")
print("Info for all dataset columns_name, dataType, null_count\n")
print(df.info())

print("\n\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n\n")
print("Size of the dataset\n")
print(df.shape)

print("\n\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n\n")
print("Description of data min, max, mean, std values for all columns\n")
print(df.describe(include='all'))

print("\n\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n\n")
print("Finding top 10 clubs with give highest average wages\n")
clubs = set(df["Club"].tolist())
dict1 = {}
dict_new = {}
for i in clubs:
    dict1[i] = []
for i in range(len(df.axes[0])):
    zval = df["Wage"].tolist()[i]
    if '€' in zval: zval = zval[1:]
    if 'K' in zval: zval = float(zval[:-1])*1000
    elif 'M' in zval:
        zval = zval[:-1]
        zval = float(zval)*1000000
    dict1[df["Club"].tolist()[i]].append(float(zval))
for i in clubs:
    dict_new[i] = np.average(np.array(dict1[i]))
dict_new = sorted(dict_new.items(), key = lambda kv:(kv[1], kv[0]), reverse=1) 
k = collections.Counter(dict_new)
high = k.most_common(10)
club_name = []
club_wage = []
for i in high:
    club_name.append(i[0][0])
    club_wage.append(i[0][1])
matplotlib.pyplot.figure(figsize=(15,7))
matplotlib.pyplot.bar(club_name,club_wage)
matplotlib.pyplot.xticks(rotation=70)
matplotlib.pyplot.title("Clubs with highest wages", color="black")
matplotlib.pyplot.show()

print("\n\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n\n")
print("Finding top 10 countries with highest number of players\n")
clubs = set(df["Nationality"].tolist())
dict1 = {}
dict_new = {}
for i in clubs:
    dict1[i] = []
for i in range(len(df.axes[0])):
    zval = df["Overall"].tolist()[i]
    dict1[df["Nationality"].tolist()[i]].append(float(zval))
for i in clubs:
    dict_new[i] = len(dict1[i])
dict_new = sorted(dict_new.items(), key = lambda kv:(kv[1], kv[0]), reverse=1) 
k = collections.Counter(dict_new)
high = k.most_common(10)
club_name = []
club_wage = []
for i in high:
    club_name.append(i[0][0])
    club_wage.append(i[0][1])
matplotlib.pyplot.figure(figsize=(15,7))
matplotlib.pyplot.bar(club_name,club_wage)
matplotlib.pyplot.xticks(rotation=70)
matplotlib.pyplot.title("Top 10 countries with most number of players", color="black")
matplotlib.pyplot.show()

print("\n\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n\n")
print("Finding frequency of heights")
X = df['Height'].tolist()
new = []
for i in X:
    if "'" in str(i):
        hh = i.split("'")
        hhh = int(hh[0])*12 + int(hh[1])
        new.append(hhh)
matplotlib.pyplot.hist(new,bins=[65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81])
matplotlib.pyplot.xlabel("height")
matplotlib.pyplot.ylabel('freq')
matplotlib.pyplot.show()

print("\n\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n\n")
print("Finding distribution of heights vs. weights\n")
Y = df['Weight'].tolist()
newY = []
for i in Y:
    if "lbs" in str(i):
        hh = i.split("l")
        hhh = int(hh[0])
        newY.append(hhh)
matplotlib.pyplot.scatter(new,newY)
matplotlib.pyplot.xlabel('Height')
matplotlib.pyplot.ylabel('Weight')
matplotlib.pyplot.show()

print("\n\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n\n")
print("Finding frequency of different ages\n")
agenum = df[['Age',"ID"]].groupby(by=['Age'],as_index=0).count().sort_values("ID",ascending=0)
'''
Getting age
'''
agenum = agenum.reset_index().drop(["index"], axis=1)
agenum.head()
'''
plotting these against frequency
'''
_,bins,_ = matplotlib.pyplot.hist(df.Age,bins=df.Age.max()-df.Age.min()+1,label="Age with count")
v1, v2 = scipy.stats.norm.fit(df.Age)
ggez = bins
best_fit_line = scipy.stats.norm.pdf(bins, v1, v2)
demn = v1
# print(ggez, demn)
matplotlib.pyplot.plot(bins, df.shape[0] * best_fit_line,label="fit_line",color="orange")
matplotlib.pyplot.title('Distrbution of Age with players')
matplotlib.pyplot.ylabel("Count of players")
matplotlib.pyplot.xlabel("Age")
matplotlib.pyplot.legend()
matplotlib.pyplot.show()


print("\n\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n\n")
print("Getting pie chart for count of players with right and left foot preferences\n")
which = df.groupby("Preferred Foot")["Preferred Foot"].count()
print(which)
'''
plot pie chart to display the percentage for every foot that players preferred
'''
matplotlib.pyplot.pie(which, labels=["Left footed","Right footed"])
matplotlib.pyplot.legend()
matplotlib.pyplot.show()


print("\n\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n\n")
print("Number of players for each position\n")
'''
count number for every position in playground that have players and sorted it.
'''
ppos = df[["Position","ID"]].groupby(by=['Position'],as_index=0).count().sort_values("ID",ascending=0)
ppos = ppos.reset_index().drop(["index"], axis=1)
ppos.head()
'''
plot bar chart to display the number of players for every position with sorted.
'''
matplotlib.pyplot.figure(figsize=(16,9))
matplotlib.pyplot.bar(ppos["Position"],ppos["ID"])
matplotlib.pyplot.title("Player's Position Distrbution", color="black")
matplotlib.pyplot.show()


print("\n\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n\n")
print("Getting top 10 forwards and right forwards\n")
'''
get top 10 ST players in the world.
'''
stpos = df[df["Position"]=="ST"].sort_values("Overall",ascending=0)[["Name","Overall"]]
stpos = stpos.iloc[:5,:]
rfpos = df[df["Position"]=="RF"].sort_values("Overall",ascending=0)[["Name","Overall"]]
rfpos = rfpos.iloc[:5,:]
'''
function plot bar chart for top 10 player in selected position.
'''
fig, axes = matplotlib.pyplot.subplots(nrows=1, ncols=2, figsize=[16, 9])
seaborn.barplot(stpos["Name"],stpos["Overall"] , ax=axes[1]).set_title("Top 5 " + "ST" +" players")
seaborn.barplot(rfpos["Name"],rfpos["Overall"] , ax=axes[0]).set_title("Top 5 " + "RF" +" players")
matplotlib.pyplot.show()


print("\n\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n\n")
print("Plotting histogram for overall attribute\n")
'''
plot the distribution of overall rating.
'''
matplotlib.pyplot.figure(figsize=(16, 9))
seaborn.countplot(df.Overall, label="overall_hist")
matplotlib.pyplot.title("Overall rating distribution for all Player")
matplotlib.pyplot.show()


print("\n\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n\n")
print("Plotly graphs for certain players\n")
shootarr =  ["Positioning","Finishing","ShotPower","LongShots","Volleys","Penalties"]
passarr = ["Vision","Crossing","FKAccuracy","ShortPassing","LongPassing","Curve"]
dribblearr = ["Agility","Balance", "Reactions", "BallControl", "Dribbling","Composure"]
defendarr = ["Interceptions", "HeadingAccuracy", "Marking", "StandingTackle","SlidingTackle"]
phyarr = ["Jumping", "Stamina", "Strength","Aggression"]
attribute_dict = {"shooting" : shootarr,
                  "passing" : passarr,
                  "dribbling" : dribblearr,
                  "defending" : defendarr,
                  "physical" :phyarr }

'''
funcion that calcualte attribute for any player: need using player index
'''
def calatt(df,indx):
    cols = []
    for i in attribute_dict.values():
        cols.extend(i) 
    obs = df.loc[indx,cols].astype("int64")
    skills = []
    for i in attribute_dict.keys():
        lis = attribute_dict.get(i)
        sumobs = sum(obs[lis])
        lenobs = len(obs[lis])
        skills.append(int(sumobs/lenobs))
    ret  = {a.upper()+": "+str(b)+"%":b for a,b in zip(attribute_dict.keys(),skills)}
    return ret
def ppr(skills,name):
    fig = plotly.graph_objects.Figure()
    fig.add_trace(plotly.graph_objects.Scatterpolar(r=list(skills.values()),theta=list(skills.keys()),fill='toself',
                                  name=name,))
    fig.update_layout(polar=dict(radialaxis=dict(visible=False,range=[0,100])),showlegend=True)
    fig.show()

# MESSI RONALDO NEYMAR
for i in range(3):
    ppr(calatt(df,i),df.iloc[i]["Name"])


print("\n\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n\n")
print("Finding best team (DREAM 11)\n")

'''
player position in each line in playground.
'''

deff = ["RB","LB","CB","LCB","RCB","RWB","RDM","CDM","LDM","LWB"]
midd = ["RM","LM","CM","LCM","RCM","RAM","CAM","LAM"]
att = ["RW","RF","CF","LF","LW","RS","ST","LS"]
position = {"deffender": deff,
             "midder": midd,
             "attacker": att
            }


'''
function that get best squad in the world based on your Lineup.
'''

dream11 = df[df.Position == "GK"].sort_values("Overall",ascending=0).iloc[0:1]
for j, k in zip(position.keys(),range(3)):
    lineup = [3,4,3]
    best = []
    for i in position.get(j):
        best.append(df[df.Position == i].sort_values(["Overall","Potential"], ascending=[0,0]).iloc[0])
    best = pd.DataFrame(best).sort_values(["Overall","Potential"], ascending=[0,0])
    best = best.iloc[0:lineup[k]]
    dream11 = pd.concat([dream11, best])

'''
get best squad on the world based on lineup which you select.
'''
dream11.reset_index(inplace=True)
player_index = list(dream11.loc[:,["index"]].values.reshape(11,))
dream11.drop("index",axis=1,inplace=True)
print(dream11)

print("\n\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n\n")
print("Getting scatter plots for value, overall and wage\n")
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
matplotlib.pyplot.scatter(Ynew,Xnew)
matplotlib.pyplot.xlabel('Overall')
matplotlib.pyplot.ylabel('Value')
matplotlib.pyplot.show()
matplotlib.pyplot.scatter(Ynew,Znew)
matplotlib.pyplot.xlabel('Overall')
matplotlib.pyplot.ylabel('Wage')
matplotlib.pyplot.show()
matplotlib.pyplot.scatter(Xnew,Znew)
matplotlib.pyplot.xlabel('Value')
matplotlib.pyplot.ylabel('Wage')
matplotlib.pyplot.show()

print("\n\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n\n")
print("End of visualization\n")
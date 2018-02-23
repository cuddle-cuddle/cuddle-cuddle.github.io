
## Still not NLP, but some modeling

The purpose of this is to shed light on the jokes data we have retrieved.
Trump is a popular butt end of a joke on /r/Jokes. Are there any patterns in this? Doest the election have any result on his un-popularity?

And answer to many more quesitons.
-- any other event that makes people hate trump
-- how quickly do people gain/lose intersting in trump bashing
-- etc. etc.

## Data import and pre-process


```python
import pandas as pd
df = pd.read_csv('./jokes_score_name_clean.csv')
```


```python
name_list = ["hilary", "clinton", "obama", "bush", "trump", "biden", "cheney", "ajit", "mccain", "palin"]
df.sample(5)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>id</th>
      <th>score</th>
      <th>q</th>
      <th>a</th>
      <th>timestamp</th>
      <th>name</th>
      <th>hilary</th>
      <th>clinton</th>
      <th>obama</th>
      <th>bush</th>
      <th>trump</th>
      <th>biden</th>
      <th>cheney</th>
      <th>ajit</th>
      <th>mccain</th>
      <th>palin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>69808</th>
      <td>69822</td>
      <td>5i3cbe</td>
      <td>24</td>
      <td>Selling a dead bird</td>
      <td>Not going cheep</td>
      <td>1.481633e+09</td>
      <td>Ibrahhhhh</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>121640</th>
      <td>121657</td>
      <td>7e0tgh</td>
      <td>212</td>
      <td>I hope I never meet Frank</td>
      <td>Every time someone tries to be Frank with me t...</td>
      <td>1.511101e+09</td>
      <td>Electricboogalou</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>65203</th>
      <td>65217</td>
      <td>5ci2mj</td>
      <td>10</td>
      <td>What do you call a promise you can't keep?</td>
      <td>A campaign promise.</td>
      <td>1.478913e+09</td>
      <td>NicCage4life</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>15992</th>
      <td>16001</td>
      <td>3f1t0p</td>
      <td>5</td>
      <td>How many people does it take to screw in a lig...</td>
      <td>Only two, but either they'd have to be really ...</td>
      <td>1.438190e+09</td>
      <td>metagloria</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>130108</th>
      <td>130125</td>
      <td>7sc4c1</td>
      <td>70</td>
      <td>How do you spot a blind guy on a nude beach?</td>
      <td>It isn't hard.</td>
      <td>1.516684e+09</td>
      <td>gmb263</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
from datetime import datetime, date
import time
def toDateStr(t):
    s = datetime.fromtimestamp(t)
    return s.strftime('%Y-%m-%d')

def timeToStamp(s):
    return time.mktime(s.timetuple())
```


```python
df.sort_index='timestamp'
```


```python
df.sort_values(by='timestamp', inplace=True)
```


```python
dflen = len(df.index)
currtotal = 0
totals = []
for i in range(dflen):
    row = df.iloc[i]
    currtotal = currtotal +row.score
    totals.append(currtotal)
```

The reason why we're adding all the scores together is that this is a dumb way to do integration.
ideally, we want to know how many posts/score about trumps are posted per hour/day, but the data is far from smooth.
Thus, the poopr person's integration.


```python
df['sum_score'] = totals
```

we're also taking advantage of the matplotlib's plot_date function, so that the x axis is labeled nicely for us.


```python
df['date'] = df['timestamp'].apply(datetime.fromtimestamp)
```

# Experiment 1: poke around, see what happens


```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
```

## Preliminary look: We started making way more fun of Trump after the election, Nov. 8th, 2016
### The inflection point is pretty fucking hard to miss.


```python
plt.clf()
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 10)
plt.xlabel('Time')
plt.ylabel('Total Upvotes')
plt.plot_date(df['date'], df['sum_score'])
plt.show()
```


![png]({{ site.baseurl }}/jupyter/trump_cancer_blog/output_16_0.png)


## Take a closer look to the election day
### a clear "jump" on Nov. 8th, 2016


```python
def getDFRange(start_date, end_date):
    return df[(df['timestamp']>= timeToStamp(start_date))
                  & (df['timestamp']< timeToStamp(end_date))]
```


```python
start_date = date(2016, 11, 1)
end_date = date(2016, 12,1)
plt.clf()
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 10)
plt.xlabel('Time')
plt.ylabel('Total Upvotes')
df_range = getDFRange(start_date, end_date)
plt.plot_date(df_range['date'], df_range['sum_score'])
plt.show()
```


![png]({{ site.baseurl }}/jupyter/trump_cancer_blog/output_19_0.png)


## But closer observation renders the data discontinuous.
... one educated guess is that Reddit derped upon the huge input of jokes, so only update /r/Jokes page(subreddit) every n minutes.


```python
start_date = date(2016, 11, 7)
end_date = date(2016, 11,14)
plt.clf()
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 10)
plt.xlabel('Time')
plt.ylabel('Total Upvotes')
df_range = getDFRange(start_date, end_date)
plt.plot_date(df_range['date'], df_range['sum_score'])
plt.show()
```


![png]({{ site.baseurl }}/jupyter/trump_cancer_blog/output_21_0.png)


### this calls for a more sophisticated way to look at the data.

# Experiment 2: binning the data per hour

### getting the trump related joke upvotes per hour and per day


```python
from datetime import timedelta, date

def hourrange(start_date, end_date):
    for n in range(int ((end_date - start_date).days*24)):
        start_of_day = (start_date + timedelta(n))
        yield time.mktime(start_of_day.timetuple())

start_date = date(2016, 1, 1)
end_date = date(2018, 1, 1)
hour_list = []

for single_date in hourrange(start_date, end_date):
    hour_list.append(single_date)

trump_data_perhour = []
for i in range(len(hour_list)-1):
    starttime = hour_list[i]
    endtime = hour_list[i+1]
    day_df = df[(df['timestamp']>= starttime) & (df['timestamp'] <endtime)]
    count = day_df[day_df['trump']>0]['score'].sum()
    trump_data_perhour.append([starttime, count])

```

Getting the data within the timeframe of intest


```python
def getDateRangeData(data, start_date, end_date):
    start_timestamp = time.mktime(start_date.timetuple())
    end_timestamp = time.mktime(end_date.timetuple())

    new_x = []
    new_y = []
    for d in data:
        if (d[0] >= start_timestamp) & (d[0]<end_timestamp):
            new_x.append(datetime.fromtimestamp(d[0]))
            new_y = new_y + [d[1]]
            #print(d)
    return new_x, new_y
```


```python
start_date = date(2016, 10, 1)
end_date = date(2017, 1, 1)

newx, newy = getDateRangeData(trump_data_perhour, start_date, end_date)
sumy = []
curry = newy[0]
for y in newy:
    sumy.append(curry)
    curry = curry + y
```


```python
plt.figure(figsize=(10,6))
plt.plot_date(newx, sumy, 'b-', label='data')
plt.legend()
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 10)
plt.xlabel('Time')
plt.ylabel('Total Upvotes')
plt.show()
```


![png]({{ site.baseurl }}/jupyter/trump_cancer_blog/output_29_0.png)


... Oh, my. What do we see here?
## A growth Curve! (around election day).
This is a very classical growth curve that's used to model many many things, from growth of bacteria to that of cancer.
COOL! Now we can model the growth of Trump jokes as if it were cancer!

# Experiment 3: Magify! Choose model! Fit Curve!


```python
start_date = date(2016, 11, 8)
end_date = date(2017, 1, 1)

newx, newy = getDateRangeData(trump_data_perhour, start_date, end_date)
sumy = []
curry = newy[0]
for y in newy:
    sumy.append(curry)
    curry = curry + y
plt.plot_date(newx, sumy)
plt.show()
```


![png]({{ site.baseurl }}/jupyter/trump_cancer_blog/output_32_0.png)


## Model: Generalized Logistic Function, with constant growth
Hum, what else does this curve remind you of?
Hint: deep learning?
Answer: Sigmoid!
Because it is but just a generalized function of sigmoid, a close cousin.



```python
# "normalize" the timestamp a bit, so it's easier to deal with.
xdata = [timeToStamp(x)/1e9 for x in newx]
ydata = sumy
```


```python
# Generalised logistic function
def growth_func(x, a, k,c,q,b, v):
    return a +(k/np.power((c + q*np.exp(b*x)), v))
```

fit curve and find coefficients:
 p0=[  9.88700433e+05,  -1.62777400e+00,  -1.02823909e+00,
         4.65014791e-01,   5.36906321e-01,   1.65061719e+00]


```python
popt, pcov = curve_fit(growth_func, xdata, ydata,
                       p0=np.array([  9.88700433e+05,  -1.62777400e+00,  -1.02823909e+00, 4.65014791e-01,   5.36906321e-01,   1.65061719e+00]),
                       maxfev=10000)
print("a, l, k, c, q, b, v: ", popt)
```

    a, l, k, c, q, b, v:  [  3.45814289e+05  -3.16163262e-06  -1.02811219e+00   4.65071816e-01
       5.36987655e-01   3.48386303e+00]


    /usr/local/lib/python3.4/dist-packages/ipykernel_launcher.py:3: RuntimeWarning: invalid value encountered in power
      This is separate from the ipykernel package so we can avoid doing imports until


## ... if this is not a perfect fucking fit, I don' tknow what is

...but I also did have 7 parameters, so over fitting is entirely possible.


```python
#map(lambda x: datetime.fromtimestamp(x*1e9), xdata)
```


```python
ys = [growth_func(*np.insert(popt, 0, d)) for d in xdata]
plt.figure(figsize=(10,6))
plt.plot_date(newx, ys, 'r-')
#         label='fit: a=%5.3f, b=%5.3f, c=%5.3f, d=%5.3f, e=%5.3f, aa=%5.3f, bb=%5.3f' % tuple(popt))
plt.plot_date(newx, ydata, 'b-', label='data')
plt.legend()
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 10)
plt.xlabel('Time')
plt.ylabel('Total Upvotes')
plt.show()
```


![png]({{ site.baseurl }}/jupyter/trump_cancer_blog/output_40_0.png)


## intepretation of parameters:

### Growth Rate = 0.54

# Experiment 4: how does it fit across a larger time frame?


```python
start_date = date(2016, 11, 8)
end_date = date(2017, 5, 8)

extended_newx, extended_newy = getDateRangeData(trump_data_perhour, start_date, end_date)
extended_sumy = []
curry = extended_newy[0]
for y in extended_newy:
    extended_sumy.append(curry)
    curry = curry + y

extended_xdata = [timeToStamp(x)/1e9 for x in extended_newx]
extended_ydata = extended_sumy
```


```python
extended_ys = [growth_func(*np.insert(popt, 0, d)) for d in extended_xdata]
```


```python
plt.figure(figsize=(10,6))
plt.plot_date(extended_newx, extended_ys, 'r-')
#         label='fit: a=%5.3f, b=%5.3f, c=%5.3f, d=%5.3f, e=%5.3f, aa=%5.3f, bb=%5.3f' % tuple(popt))
plt.plot_date(extended_newx, extended_ydata, 'b-', label='data')
plt.legend()
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 10)
plt.xlabel('Time')
plt.ylabel('Total Upvotes')
plt.show()
```


![png]({{ site.baseurl }}/jupyter/trump_cancer_blog/output_46_0.png)


# You see that point where the curve stopped fitting?
# Yep, that's the travel ban.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load the dataset 

x = pd.read_csv('./zomato.csv')

# check what all variables/fields are there in the dataset
print(x.head())

"""
                                                 url  ... listed_in(city)
0  https://www.zomato.com/bangalore/jalsa-banasha...  ...    Banashankari
1  https://www.zomato.com/bangalore/spice-elephan...  ...    Banashankari
2  https://www.zomato.com/SanchurroBangalore?cont...  ...    Banashankari
3  https://www.zomato.com/bangalore/addhuri-udupi...  ...    Banashankari
4  https://www.zomato.com/bangalore/grand-village...  ...    Banashankari

[5 rows x 17 columns]
"""

print(x.columns)

"""
Index(['url', 'address', 'name', 'online_order', 'book_table', 'rate', 'votes',
       'phone', 'location', 'rest_type', 'dish_liked', 'cuisines',
       'approx_cost(for two people)', 'reviews_list', 'menu_item',
       'listed_in(type)', 'listed_in(city)'],
      dtype='object')
"""

# get some statistics on the data
print(x.describe())
"""
              votes
count  51717.000000
mean     283.697527
std      803.838853
min        0.000000
25%        7.000000
50%       41.000000
75%      198.000000
max    16832.000000

votes is the only column with numeric values
"""

# get some infos on the dataset
print(x.info())
"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 51717 entries, 0 to 51716
Data columns (total 17 columns):
 #   Column                       Non-Null Count  Dtype
---  ------                       --------------  -----
 0   url                          51717 non-null  object
 1   address                      51717 non-null  object
 2   name                         51717 non-null  object
 3   online_order                 51717 non-null  object
 4   book_table                   51717 non-null  object
 5   rate                         43942 non-null  object
 6   votes                        51717 non-null  int64
 7   phone                        50509 non-null  object
 8   location                     51696 non-null  object
 9   rest_type                    51490 non-null  object
 10  dish_liked                   23639 non-null  object
 11  cuisines                     51672 non-null  object
 12  approx_cost(for two people)  51371 non-null  object
 13  reviews_list                 51717 non-null  object
 14  menu_item                    51717 non-null  object
 15  listed_in(type)              51717 non-null  object
 16  listed_in(city)              51717 non-null  object
dtypes: int64(1), object(16)
memory usage: 6.7+ MB


As we can observe, some columns have missing values. let's verify it
"""
print(x.isnull().sum())

"""
url                                0
address                            0
name                               0
online_order                       0
book_table                         0
rate                            7775
votes                              0
phone                           1208
location                          21
rest_type                        227
dish_liked                     28078
cuisines                          45
approx_cost(for two people)      346
reviews_list                       0
menu_item                          0
listed_in(type)                    0
listed_in(city)                    0
dtype: int64
"""

# number of unique location
unique_locations = x['location'].nunique()
print(unique_locations) # 93

# Print the unique locations
unique_location_names = x['location'].unique()

for location in unique_location_names:
    print(location)
    
"""
Banashankari
Basavanagudi
Mysore Road
Jayanagar
Kumaraswamy Layout
Rajarajeshwari Nagar
Vijay Nagar
Uttarahalli
JP Nagar
South Bangalore
City Market
Nagarbhavi
Bannerghatta Road
BTM
Kanakapura Road
Bommanahalli
nan
CV Raman Nagar
Electronic City
HSR
Marathahalli
Sarjapur Road
Wilson Garden
Shanti Nagar
Koramangala 5th Block
Koramangala 8th Block
Richmond Road
Koramangala 7th Block
Jalahalli
Koramangala 4th Block
Bellandur
Whitefield
East Bangalore
Old Airport Road
Indiranagar
Koramangala 1st Block
Frazer Town
RT Nagar
MG Road
Brigade Road
Lavelle Road
Church Street
Ulsoor
Residency Road
Shivajinagar
Infantry Road
St. Marks Road
Cunningham Road
Race Course Road
Commercial Street
Vasanth Nagar
HBR Layout
Domlur
Ejipura
Jeevan Bhima Nagar
Old Madras Road
Malleshwaram
Seshadripuram
Kammanahalli
Koramangala 6th Block
Majestic
Langford Town
Central Bangalore
Sanjay Nagar
Brookefield
ITPL Main Road, Whitefield
Varthur Main Road, Whitefield
KR Puram
Koramangala 2nd Block
Koramangala 3rd Block
Koramangala
Hosur Road
Rajajinagar
Banaswadi
North Bangalore
Nagawara
Hennur
Kalyan Nagar
New BEL Road
Jakkur
Rammurthy Nagar
Thippasandra
Kaggadasapura
Hebbal
Kengeri
Sankey Road
Sadashiv Nagar
Basaveshwara Nagar
Yeshwantpur
West Bangalore
Magadi Road
Yelahanka
Sahakara Nagar
Peenya
"""

# let's manage null values of the column prize
# Replace null values with 0 in the column 'approx_cost(for two people)'
approx_cost = 'approx_cost(for two people)' 
x[approx_cost].fillna(0, inplace=True)
print(x[approx_cost].isnull().sum()) #0

# get the average price of the food

# 1- Convert string values to numeric in a specific column
x[approx_cost] = pd.to_numeric(x[approx_cost], errors='coerce')
approx_cost_mean = x[approx_cost].mean()
print(approx_cost_mean) # 413.41238839285717

# the average cost of food for 2 is 413


# Famous neighborhood kind of food

# what is a famous neighborhood? by rate? votes?
print(x['rate'])

"""
0         4.1/5
1         4.1/5
2         3.8/5
3         3.7/5
4         3.8/5
          ...
51712    3.6 /5
51713       NaN
51714       NaN
51715    4.3 /5
51716    3.4 /5
"""

print(x['votes'])

"""
0        775
1        787
2        918
3         88
4        166
        ...
51712     27
51713      0
51714      0
51715    236
51716     13
Name: votes, Length: 51717, dtype: int64
"""

# we will use rate
# Extract the values of the "rate" column in the specified format
rate_values = x['rate'].str.extract(r'(\d+\.\d+)\s*/\s*\d+')

print(rate_values)

# change the rate column to rate_values

x['rate'] = rate_values

print(x['rate'])

"""
0        4.1
1        4.1
2        3.8
3        3.7
4        3.8
        ...
51712    3.6
51713    NaN
51714    NaN
51715    4.3
51716    3.4
Name: rate, Length: 51717, dtype: object
"""

# Change the type of values in the rate column to nullable float
x['rate'] = x['rate'].astype('Float64')

print(x['rate'])

# Compute the mean of the rate column, excluding NaN values
rate_mean = x['rate'].mean()

print(rate_mean) # 3.7004488179527253

# get the number of NaN values in this column
print(x['rate'].isna().sum()) # 10052


# determine the popular restaurants

# Sort the DataFrame by the rate column in descending order
sorted_data = x.sort_values(by='rate', ascending=False)

# Print the popular restaurants based on the rate
print(sorted_data[['name', 'rate']])

"""
                                                    name  rate
35082                     Asia Kitchen By Mainland China   4.9
37613                     Asia Kitchen By Mainland China   4.9
10879                     Asia Kitchen By Mainland China   4.9
17877                             Belgian Waffle Factory   4.9
4944                         Byg Brewski Brewing Company   4.9
...                                                  ...   ...
51644                                     Punjabi Thadka  <NA>
51675                                       Topsy Turvey  <NA>
51710                                       Topsy Turvey  <NA>
51713                           Vinod Bar And Restaurant  <NA>
51714  Plunge - Sheraton Grand Bengaluru Whitefield H...  <NA>

[51717 rows x 2 columns]
"""
# the first 20 popular restaurants
print(sorted_data[['name', 'rate']].head(20))

"""
                                 name  rate
35082  Asia Kitchen By Mainland China   4.9
37613  Asia Kitchen By Mainland China   4.9
10879  Asia Kitchen By Mainland China   4.9
17877          Belgian Waffle Factory   4.9
4944      Byg Brewski Brewing Company   4.9
47987          Belgian Waffle Factory   4.9
51042                        Flechazo   4.9
39559                    Punjab Grill   4.9
43055          Belgian Waffle Factory   4.9
37099       AB's - Absolute Barbecues   4.9
27453  Asia Kitchen By Mainland China   4.9
11745          Belgian Waffle Factory   4.9
19393       AB's - Absolute Barbecues   4.9
28403  Asia Kitchen By Mainland China   4.9
11504  Asia Kitchen By Mainland China   4.9
49170     Byg Brewski Brewing Company   4.9
18496                Milano Ice Cream   4.9
10389       AB's - Absolute Barbecues   4.9
5809           Belgian Waffle Factory   4.9
32044  Asia Kitchen By Mainland China   4.9
"""

# let's drop duplicates based on name and address
x = x.drop_duplicates(subset=['address', 'name']).reset_index().drop('index', axis=1)

sorted_data = x.sort_values(by='rate', ascending=False)
# the first 20 popular restaurants
print(sorted_data[['name', 'rate']].head(20))

"""
10324                                       Punjab Grill   4.9
2242                         Byg Brewski Brewing Company   4.9
4269                                        Punjab Grill   4.9
11642  SantÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃ...   4.9
8512                                    Milano Ice Cream   4.9
4906                           AB's - Absolute Barbecues   4.9
3918                                            Flechazo   4.9
4917                      Asia Kitchen By Mainland China   4.9
3036                              Belgian Waffle Factory   4.9
10887                          AB's - Absolute Barbecues   4.8
5916                                    House Of Commons   4.8
8302                                    The Pizza Bakery   4.8
7425                              Belgian Waffle Factory   4.8
10888                                     The Globe Grub   4.8
10719                                 O.G. Variar & Sons   4.8
11167                                  The Boozy Griffin   4.8
9014                              Belgian Waffle Factory   4.8
2087                                     The Black Pearl   4.8
5474                              Belgian Waffle Factory   4.8
1994                                Brahmin's Coffee Bar   4.8
"""
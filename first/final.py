import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load the dataset 

x = pd.read_csv('./zomato.csv')

# I - Get general Infos about the dataset


# 1 -the shape of the dataset (numbers of rows and columns)

print(x.shape) # (51717, 17)

# 2- the columns data types 
# int64 = integer numbers
# float = floating point numbers
# object = text or mixed numeric or non-numeric values
print(x.dtypes)

"""
url                            object
address                        object
name                           object
online_order                   object
book_table                     object
rate                           object
votes                           int64
phone                          object
location                       object
rest_type                      object
dish_liked                     object
cuisines                       object
approx_cost(for two people)    object
reviews_list                   object
menu_item                      object
listed_in(type)                object
listed_in(city)                object
dtype: object
"""

# 3 - display the first 15 rows of the dataset
print(x.head(15))

"""
                                                  url  ... listed_in(city)
0   https://www.zomato.com/bangalore/jalsa-banasha...  ...    Banashankari
1   https://www.zomato.com/bangalore/spice-elephan...  ...    Banashankari
2   https://www.zomato.com/SanchurroBangalore?cont...  ...    Banashankari
3   https://www.zomato.com/bangalore/addhuri-udupi...  ...    Banashankari
4   https://www.zomato.com/bangalore/grand-village...  ...    Banashankari
5   https://www.zomato.com/bangalore/timepass-dinn...  ...    Banashankari
6   https://www.zomato.com/bangalore/rosewood-inte...  ...    Banashankari
7   https://www.zomato.com/bangalore/onesta-banash...  ...    Banashankari
8   https://www.zomato.com/bangalore/penthouse-caf...  ...    Banashankari
9   https://www.zomato.com/bangalore/smacznego-ban...  ...    Banashankari
10  https://www.zomato.com/bangalore/caf%C3%A9-dow...  ...    Banashankari
11  https://www.zomato.com/bangalore/cafe-shuffle-...  ...    Banashankari
12  https://www.zomato.com/bangalore/the-coffee-sh...  ...    Banashankari
13  https://www.zomato.com/bangalore/caf-eleven-ba...  ...    Banashankari
14  https://www.zomato.com/SanchurroBangalore?cont...  ...    Banashankari

[15 rows x 17 columns]
"""

# As we can observe, some columns have missing values. let's verify it
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

# II - Exercise : brief insights of the zomato dataset
"""
1 - location of the restaurants
2 - Approximate price of food
3 - Famous neigborhood
4 - theme based restaurant or not
5 - which locality serve the food with the highest number of restaurants
6 - people needs
"""
# number of unique location
unique_locations = x['location'].nunique()
print(unique_locations) # 93

# II - 1 - location of the restaurants
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





# number of occurence of restaurants : 8792

# remove duplicates of restaurants
x = x.drop_duplicates(subset=['address', 'name']).reset_index().drop('index', axis=1)

unique_restaurant = x.groupby('name')['address'].count()
print(unique_restaurant)

"""
name
#FeelTheROLL                                    1
#L-81 Cafe                                      1
#Vibes Restro                                   1
#refuel                                         1
'Brahmins' Thatte Idli                          1
                                               ..
late100                                         1
nu.tree                                         4
re:cess - Hilton Bangalore Embassy GolfLinks    1
repEAT Hub                                      1
sCoolMeal                                       1
Name: address, Length: 8792, dtype: int64
"""

# number of unique restaurants per location 

the_list = x.groupby('location')['url'].count().sort_values(ascending=False)
ax = the_list.plot(kind='bar', figsize=(15, 8), rot=90, width = 0.5, color=[ 'pink'])
rects = ax.patches
labels = list(the_list)
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2, height + .05, label,
            ha='center', va='bottom', fontsize=7)
ax.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on') # remove borders
ax.xaxis.set_tick_params(labelsize=5) # 
ax.legend(fontsize=14) # set legend sie as 14
ax.set_title('No of restaurants', fontsize=6) # set title and add font size as 16
ax.set_xlabel('Neighborhood', fontsize=6)
#ax.grid(False)  # remove grid
ax.set_facecolor("white") # set bg color white
ax.legend(['#Restaurants'])
plt.show()


# ( les 30 premières locations)
the_list = x.groupby('location')['url'].count().sort_values(ascending=False)[:30]
ax = the_list.plot(kind='bar', figsize=(15, 8), rot=90, width = 0.5, color=[ 'pink'])
rects = ax.patches
labels = list(the_list)
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2, height + .05, label,
            ha='center', va='bottom', fontsize=7)
ax.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on') # remove borders
ax.xaxis.set_tick_params(labelsize=5) # 
ax.legend(fontsize=14) # set legend sie as 14
ax.set_title('No of restaurants', fontsize=6) # set title and add font size as 16
ax.set_xlabel('Neighborhood', fontsize=6)
#ax.grid(False)  # remove grid
ax.set_facecolor("white") # set bg color white
ax.legend(['#Restaurants'])
plt.show()

# II - 2 - Approximate price of food

# price of food for 2 in each neigborhood

approx_cost = 'approx_cost(for two people)' 
x[approx_cost] = x[approx_cost].str.replace(",","").astype(float)

my_list = x.groupby('location')[approx_cost].mean().sort_values(ascending= False)

print(my_list)

"""
location
Sankey Road         2526.923077
Lavelle Road        1323.214286
Race Course Road    1316.666667
MG Road             1056.122449
Infantry Road        966.666667
                       ...
Peenya               300.000000
Yelahanka            300.000000
City Market          299.977778
Shivajinagar         298.360656
North Bangalore      291.666667
Name: approx_cost(for two people), Length: 93, dtype: float64
"""

# The average price of the food accross all locations
x[approx_cost] = pd.to_numeric(x[approx_cost], errors='coerce')
approx_cost_mean = x[approx_cost].mean()
print(approx_cost_mean) # 487.2

# Plot the approximate price of food per location
plt.figure(figsize=(12, 6))
x.groupby('location')[approx_cost].mean().sort_values().head(30).plot(kind='bar', color=['pink'])
plt.title('Approximate Price of Food per Location')
plt.xlabel('Location')
plt.ylabel('Average Approximate Cost for 2 people')
plt.xticks(rotation=45)
plt.show()

# II - 3- Famous neigborhood ( by rate)

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
12494    NaN
12495    4.2
12496    3.3
12497    2.5
12498    NaN
Name: rate, Length: 12499, dtype: object
"""
# Change the type of values in the rate column to nullable float
x['rate'] = x['rate'].astype('Float64')

# now, print the location by ratings
num_rate_x = x.groupby(['location'])['rate'].first()
num_rate_x['rate'] = x.groupby(['location'])['rate'].mean()

print(num_rate_x)
"""
BTM                                                                3.0
Banashankari                                                       4.1
Banaswadi                                                          3.4
Bannerghatta Road                                                  4.4
Basavanagudi                                                       3.8
                                           ...
Whitefield                                                         3.9
Wilson Garden                                                      3.8
Yelahanka                                                          3.5
Yeshwantpur                                                        3.4
"""

# II - 4 - theme based restaurant or not

# types of restaurants

print(x['rest_type'])

"""
0              Casual Dining
1              Casual Dining
2        Cafe, Casual Dining
3                Quick Bites
4              Casual Dining
                ...
12494            Quick Bites
12495          Casual Dining
12496            Quick Bites
12497     Casual Dining, Bar
12498                    Bar
Name: rest_type, Length: 12499, dtype: object
"""

# list of uniques restaurants types

types = set()
def func(x):
    if(type(x) == list):
        for y in x:
            types.add(y.strip())
_ = x['rest_type'].str.split(',').apply(func)

restaurant_types = list(types)
print(restaurant_types) # 25 types of restaurants

"""
['Cafe', 'Bakery', 'Sweet Shop', 'Takeaway', 'Food Court', 'Bhojanalya', 'Dhaba',
'Food Truck', 'Lounge', 'Dessert Parlor', 'Irani Cafee', 'Delivery', 'Casual Dining',
'Beverage Shop', 'Quick Bites', 'Confectionery', 'Microbrewery', 'Mess', 'Kiosk', 'Club',
'Meat Shop', 'Bar', 'Fine Dining', 'Pub', 'Pop Up']
"""

location_types = x.groupby(['location'])['rest_type'].first()
print(location_types)

"""
BTM                  Casual Dining
Banashankari         Casual Dining
Banaswadi                     Cafe
Bannerghatta Road              Pub
Basavanagudi         Casual Dining
                         ...
West Bangalore            Delivery
Whitefield             Quick Bites
Wilson Garden        Casual Dining
Yelahanka              Quick Bites
Yeshwantpur          Casual Dining
Name: rest_type, Length: 93, dtype: object

"""
restaurant_types = x.groupby('location')['rest_type'].unique()

# for location, types in restaurant_types.items():
#     print(f"Location: {location}")
#     print(f"Restaurant Types: {types}")
#     print()
    



print(restaurant_types)   

"""
BTM                  [Casual Dining, Cafe, Delivery, Quick Bites, D...
Banashankari         [Casual Dining, Cafe, Casual Dining, Quick Bit...
Banaswadi            [Cafe, Quick Bites, Delivery, Casual Dining, S...
Bannerghatta Road    [Pub, Casual Dining, Cafe, Food Court, Cafe, B...
Basavanagudi         [Casual Dining, Casual Dining, Cafe, Cafe, Qui...
                                           ...
West Bangalore              [Delivery, Takeaway, Delivery, Food Truck]
Whitefield           [Quick Bites, Casual Dining, Casual Dining, Ba...
Wilson Garden        [Casual Dining, Quick Bites, Delivery, Cafe, S...
Yelahanka                                                [Quick Bites]
Yeshwantpur          [Casual Dining, Quick Bites, Sweet Shop, Quick...
Name: rest_type, Length: 93, dtype: object

"""

# Count the number of restaurant types per location
type_counts = x.groupby('location')['rest_type'].nunique()


# Select the first 30 locations
type_counts = type_counts.head(30)

# Plot the bar chart
plt.figure(figsize=(10, 6))
type_counts.plot(kind='bar', color='pink')
plt.xlabel('Location')
plt.ylabel('Number of uniques Restaurant Types')
plt.title('Number of Restaurant Types per Location')
plt.xticks(rotation=45)
plt.show()

# let's do the same for the type of cuisines

type_counts = x.groupby('location')['cuisines'].nunique()


# Select the first 30 locations
type_counts = type_counts.head(30)

# Plot the bar chart
plt.figure(figsize=(10, 6))
type_counts.plot(kind='bar', color='pink')
plt.xlabel('Location')
plt.ylabel('Number of  cuisines')
plt.title('Number of Cuisines per Location')
plt.xticks(rotation=45)
plt.show()

# II - 5 - which locality serve the food with the highest number of restaurants

# find the location with the maximum count of unique restaurant types.

# Count the number of restaurant types per location
type_counts = x.groupby('location')['rest_type'].nunique()

# Find the location with the highest number of restaurant types
location_with_max_restaurants = type_counts.idxmax()

# Print the location with the highest number of restaurant types
print(f"The location with the highest number of restaurants is: {location_with_max_restaurants}")

# The location with the highest number of restaurants is: Whitefield

# now let's print the first 10 locations with the highest number of restaurants
# use the nlargest() function from pandas
# Get the first 10 locations with the highest number of restaurants
top_10_locations = type_counts.nlargest(10)

# Print the first 10 locations with the highest number of restaurants
print("Top 10 locations with the highest number of restaurants:")
for location, count in top_10_locations.items():
    print(f"Location: {location}, Number of Restaurants: {count}")
    
"""
Top 10 locations with the highest number of restaurants:
Location: Whitefield, Number of Restaurants: 47
Location: HSR, Number of Restaurants: 40
Location: Indiranagar, Number of Restaurants: 40
Location: Bellandur, Number of Restaurants: 38
Location: JP Nagar, Number of Restaurants: 37
Location: Marathahalli, Number of Restaurants: 37
Location: Malleshwaram, Number of Restaurants: 34
Location: Electronic City, Number of Restaurants: 33
Location: Koramangala 5th Block, Number of Restaurants: 33
Location: Bannerghatta Road, Number of Restaurants: 31
"""

# plot the locations with the number of cuisines ordered 

# Count the number of unique cuisines ordered per location
cuisine_counts = x.groupby('location')['cuisines'].nunique()

#cuisine_counts = cuisine_counts.head(30)

# Plot the bar chart
plt.figure(figsize=(10, 6))
cuisine_counts.plot(kind='bar', color='pink')
plt.xlabel('Location')
plt.ylabel('Number of Unique Cuisines')
plt.title('Number of Unique Cuisines per Location')
plt.xticks(rotation=45)
plt.show()




# II - 6 - people needs


# top 10 cuisines of the dataset 

# Count the occurrences of each cuisine
cuisine_counts = x['cuisines'].value_counts().head(10)

# Plot the bar chart
plt.figure(figsize=(10, 6))
cuisine_counts.plot(kind='bar', color='pink')
plt.xlabel('Cuisine')
plt.ylabel('Count')
plt.title('Top 10 Cuisines')
plt.xticks(rotation=45)
plt.show()


# top rated cuisines

# Group the data by cuisine and calculate the average rating
average_rating = x.groupby('cuisines')['rate'].mean()

# Sort cuisines by average rating in descending order
top_rated_cuisines = average_rating.sort_values(ascending=False).head(10)

# Plot the bar chart
plt.figure(figsize=(10, 6))
top_rated_cuisines.plot(kind='bar', color='pink')
plt.xlabel('Cuisine')
plt.ylabel('Average Rating')
plt.title('Top Rated Cuisines')
plt.xticks(rotation=45)
plt.show()

# let determines the top rated cuisines and the locations where to find them

# Get the locations for the top rated cuisines
top_rated_locations = x[x['cuisines'].isin(top_rated_cuisines.index)][['cuisines', 'location']]

# Print the top rated cuisines and their corresponding locations
print("Top Rated Cuisines and Their Locations:")
for cuisine in top_rated_cuisines.index:
    locations = top_rated_locations[top_rated_locations['cuisines'] == cuisine]['location'].unique()
    print(f"Cuisine: {cuisine}")
    print("Locations:", ", ".join(locations))
    print()
    
    
"""
Top Rated Cuisines and Their Locations:
Cuisine: Continental, North Indian, Italian, South Indian, Finger Food
Locations: Sarjapur Road

Cuisine: Healthy Food, Salad, Mediterranean
Locations: Indiranagar

Cuisine: Asian, Chinese, Thai, Momos
Locations: Koramangala 5th Block

Cuisine: Asian, Mediterranean, North Indian, BBQ
Locations: Whitefield, Marathahalli

Cuisine: North Indian, European, Mediterranean, BBQ
Locations: Marathahalli

Cuisine: European, Mediterranean, North Indian, BBQ
Locations: Sarjapur Road, Whitefield, BTM, Kalyan Nagar, Marathahalli

Cuisine: Continental, North Indian, BBQ, Steak
Locations: Marathahalli

Cuisine: Italian, American, Pizza
Locations: Indiranagar

Cuisine: Continental, European, BBQ, Chinese, Asian
Locations: Whitefield

Cuisine: Continental, BBQ, Salad
Locations: Residency Road
"""

# for each cuisine, determine the list of locations where to find them

# Group the data by cuisine and get the unique locations for each cuisine
cuisine_locations = x.groupby('cuisines')['location'].unique()

# Print the list of locations for each cuisine
# for cuisine, locations in cuisine_locations.items():
#     print(f"Cuisine: {cuisine}")
#     print("Locations:", ", ".join(locations))
#     print()


print(cuisine_locations)

"""
cuisines
African, Burger                                                                        [Bannerghatta Road, Whitefield]
American                                                             [Sarjapur Road, Whitefield, Indiranagar, Kalya...
American, Asian, Continental, North Indian, South Indian, Chinese                                      [Sarjapur Road]
American, Asian, European, North Indian                                                                  [Indiranagar]
American, BBQ                                                                                         [St. Marks Road]
                                                                                           ...
Turkish, Fast Food, Biryani, Chinese                                                                        [RT Nagar]
Turkish, Rolls                                                            [BTM, Jayanagar, Koramangala 5th Block, HSR]
Vietnamese                                                           [Banashankari, Koramangala 4th Block, Race Cou...
Vietnamese, Salad                                                                                        [Indiranagar]
Vietnamese, Thai, Burmese, Japanese                                                                     [Malleshwaram]
Name: location, Length: 2609, dtype: object
"""

# get the top rated restaurant type

average_rating = x.groupby('rest_type')['rate'].mean()

# Get the top rated restaurant type
top_rated_restaurant_type = average_rating.idxmax()

# Print the top rated restaurant type
print("Top Rated Restaurant Type:", top_rated_restaurant_type)

# Top Rated Restaurant Type: Bar, Pub


# Sort the restaurant types by average rating in descending order
top_10_restaurant_types = average_rating.sort_values(ascending=False).head(10)

# Print the top 10 restaurant types
print("Top 10 Restaurant Types:")
for restaurant_type, rating in top_10_restaurant_types.items():
    print(f"Restaurant Type: {restaurant_type}, Average Rating: {rating}")
    
"""
Top 10 Restaurant Types:
Restaurant Type: Bar, Pub, Average Rating: 4.6
Restaurant Type: Pub, Cafe, Average Rating: 4.550000000000001
Restaurant Type: Microbrewery, Average Rating: 4.5200000000000005
Restaurant Type: Microbrewery, Pub, Average Rating: 4.4875
Restaurant Type: Cafe, Lounge, Average Rating: 4.4
Restaurant Type: Microbrewery, Bar, Average Rating: 4.4
Restaurant Type: Fine Dining, Lounge, Average Rating: 4.4
Restaurant Type: Casual Dining, Irani Cafee, Average Rating: 4.4
Restaurant Type: Pub, Microbrewery, Average Rating: 4.4
Restaurant Type: Microbrewery, Casual Dining, Average Rating: 4.36
"""
    

# Sort cuisines by average rating in descending order and get the top 10
top_10_cuisines = average_rating.sort_values(ascending=False).head(10)

# Create a bar plot for the top 10 cuisines
plt.figure(figsize=(10, 6))
top_10_cuisines.plot(kind='bar', color='pink')
plt.xlabel('Cuisine')
plt.ylabel('Average Rating')
plt.title('Top 10 Best Rated Cuisines')
plt.xticks(rotation=45)
plt.show()


# Sort the data by rating in descending order and get the top 10 restaurants
top_20_restaurants = x.sort_values(by='rate', ascending=False).head(20)

# Create a pie chart for the top 10 rated restaurants
plt.figure(figsize=(8, 8))
plt.pie(top_20_restaurants['rate'], labels=top_20_restaurants['name'], autopct='%1.1f%%')
plt.title('Top 20 Rated Restaurants')
plt.axis('equal')
plt.show()

# visualize the top 10 rated restaurant type

# Group the data by restaurant type and calculate the average rating
average_rating = x.groupby('rest_type')['rate'].mean()

# Sort restaurant types by average rating in descending order and get the top 10
top_10_restaurant_types = average_rating.sort_values(ascending=False).head(10)

# Create a pie chart for the top 10 rated restaurant types
plt.figure(figsize=(8, 8))
plt.pie(top_10_restaurant_types, labels=top_10_restaurant_types.index, autopct='%1.1f%%')
plt.title('Top 10 Rated Restaurant Types')
plt.axis('equal')
plt.show()



# for each top 10 rated cuisines, let's plot the 10 locations where to find them

# Group the data by cuisine and calculate the average rating
average_rating = x.groupby('cuisines')['rate'].mean()

# Sort cuisines by average rating in descending order and get the top 10 rated cuisines
top_10_rated_cuisines = average_rating.sort_values(ascending=False).head(10)

# Create subplots for each top 10 rated cuisine
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(12, 18))

# Plot the top 10 locations for each cuisine
for ax, cuisine in zip(axes.flatten(), top_10_rated_cuisines.index):
    cuisine_locations = x[x['cuisines'] == cuisine]['location'].value_counts().head(10)
    
    ax.bar(cuisine_locations.index, cuisine_locations.values, color='pink')
    ax.set_title(f"Top 10 Locations for {cuisine}")
    ax.set_xlabel('Location')
    ax.set_ylabel('Count')
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
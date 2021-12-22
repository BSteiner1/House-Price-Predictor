import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\bbste\Documents\Coding\Python\Machine Learning\Melbourne Housing Data.csv")

delete_columns = ['Address', 'SellerG', 'Method', 'Date', 'Postcode', 'YearBuilt', 'Type', 'Lattitude', 'Longtitude', 'Regionname', 'Suburb', 'CouncilArea']
for i in range(len(delete_columns)):   
    df.pop(delete_columns[i])

print(df.isnull().sum())

df_heat = df.corr()
plt.figure(figsize=(9,5))
heatmap = sns.heatmap(df_heat, annot=True, cmap='coolwarm')
plt.show()

print("Original DataFrame shape: ", df.shape)

#Fill or remove NaN
#Remove variable (low correlation)
df.pop('BuildingArea')

#Fill variables with mean 
df['Car'].fillna(df['Car'].mean(),inplace=True)
df['Landsize'].fillna(df['Landsize'].mean(),inplace=True)

#Drop any remaining missing values
df.dropna(axis=0, how='any', inplace=True) 

print("New DataFrame shape: ",df.shape)

#Set variables
x = df[['Rooms', 'Distance', 'Bedroom2', 'Bathroom', 'Car', 'Landsize', 'Propertycount']]
y = df['Price']

#Divide the data into training and testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=10, shuffle=True)

#Set our model
model = LinearRegression()

#Train model with training data
model.fit(x_train.values, y_train)

print("Predict House Price")
room_input = int(input("Number of rooms: "))
distance_input = int(input("Distance: "))
bedroom2_input = int(input("Number of bedrooms: "))
bathroom_input = int(input("Number of bathrooms: "))
car_input =int(input("Number of car spaces: "))
landsize_input = int(input("Area: "))
propertycount_input = int(input("Property count: "))

new_house = [
    room_input, #Rooms
    distance_input, #Distance
    bedroom2_input, #Bedroom2
    bathroom_input, #Bathroom
    car_input, #Car
    landsize_input, #Landsize
    propertycount_input #Propertycount
]

house_prediction = model.predict([new_house])
print("House Price Prediction: ", np.round(house_prediction, 2)) 



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial

url = 'https://cdn.intra.42.fr/document/document/18562/data.csv'
#Importing a CSV file into the DataFrame
data = pd.read_csv(url, sep=",", usecols= ['km','price'])
print (data)

# plotting a bar graph
#data.plot(x="km", y="price", kind="bar")
#plt.hist(data["km"])

"""
# plotting a scatter plot
print("Scatter Plot: data.csv  ")
plt.scatter(data["km"], data["price"], c ='orange')
plt.xlabel('km')
plt.ylabel('price')
plt.title('Price to mileage')
plt.show()
"""

# convert dataframe to numpy array
arr = data[['km', 'price']].to_numpy()
 
print('\nNumpy Array\n----------\n', arr)

x = data['km'].to_numpy
y = data['price'].to_numpy
coeffs = np.polyfit(arr[0], arr[1], 1)
print('\x1b[6;30;43m' + 'COEFFICIENTS' + '\x1b[0m')
print (coeffs)
mymodel =  np.poly1d(coeffs)
myline = np.linspace(min(arr[0]),max(arr[0]),100).astype(int)

"""
print('\x1b[6;30;42m' + 'xn from np.linspace' + '\x1b[0m')
#Number of samples to generate.
myline = np.linspace(min(arr[0]),max(arr[0]),100).astype(int)
print(myline)
yn = coeffs[1] + coeffs[0] * myline.astype(float)
print('\x1b[6;30;42m' + 'estimated price' + '\x1b[0m')
np.set_printoptions(precision = 2)
print(yn)
"""

plt.scatter(data["km"], data["price"], c ='orange')
plt.plot(myline, mymodel(myline))
plt.show()

"""
np.polyval(coeffs, x)
fig, ax = plt.subplots()
ax.plot(x, y, label='data')
ax.plot(np.polyval(coeffs, xn), label='fit')
ax.legend()
"""
val = input('\x1b[6;30;43m' + 'Enter a car mileage:' + '\x1b[0m\n')
print('\x1b[1;30;42m' + 'Estimated Price ($):' + '\x1b[0m')
print ('{:.2f}'.format(coeffs[1] + float(val) * coeffs[0]))

"""
https://numpy.org/doc/stable/reference/generated/numpy.linspace.html
https://matplotlib.org/
https://seaborn.pydata.org/
https://en.wikipedia.org/wiki/ANSI_escape_code#Colors
https://stackoverflow.com/questions/287871/how-do-i-print-colored-text-to-the-terminal
"""
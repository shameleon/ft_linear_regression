import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

url = 'https://cdn.intra.42.fr/document/document/11434/data.csv'
#Importing a CSV file into the DataFrame
data = pd.read_csv(url, sep=",", usecols= ['km','price'])
print (data)

# plotting a bar graph
#data.plot(x="km", y="price", kind="bar")
#plt.hist(data["km"])

# plotting a scatter plot
print("Scatter Plot: data.csv  ")
plt.scatter(data["km"], data["price"], c ='orange')
plt.xlabel('km')
plt.ylabel('price')
plt.title('Price to mileage')
plt.show()

# convert dataframe to numpy array
arr = data[['km', 'price']].to_numpy()
 
print('\nNumpy Array\n----------\n', arr)

x = data['km'].to_numpy
y = data['price'].to_numpy
coeffs = np.polyfit(arr[0], arr[1], 1)
print('\x1b[6;30;43m' + 'COEFFICIENTS' + '\x1b[0m')
print (coeffs)
print('\x1b[6;30;42m' + 'x' + '\x1b[0m')
print (x)
xn = np.linspace(min(arr[0]),max(arr[0]),100).astype(float)
yn = coeffs[0] + coeffs[1] * xn
#plt.plot(xn, yn(xn), x, y, 'o')

val = input('\x1b[6;30;43m' + 'Enter a car mileage:' + '\x1b[0m\n')
print('\x1b[1;30;42m' + 'Estimated Price :' + '\x1b[0m')
print (coeffs[0] + float(val) * coeffs[1])

"""
https://en.wikipedia.org/wiki/ANSI_escape_code#Colors
https://stackoverflow.com/questions/287871/how-do-i-print-colored-text-to-the-terminal
"""
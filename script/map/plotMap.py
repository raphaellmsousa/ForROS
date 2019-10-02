import csv
import matplotlib.pyplot as plt

with open('map_log.csv', mode='r') as data:
	latitudeAndLongitude = csv.reader(data, delimiter=',')
	latitudes = []
	longitudes = []
	for row in latitudeAndLongitude:
		latitudes.append(row[0])
		longitudes.append(row[1])

plt.plot(-38.44402313232422, 2.393688678741455,  color = 'r', marker="P")
plt.plot(latitudes, longitudes, 'b')
#plt.xlabel('Longitude')
#plt.ylabel('Latitude')
plt.show()

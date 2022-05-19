import requests

api_key = 'AjH2UXENFQRLK2RYNaLh5yMK6lbVNIt9RVUkssIzONIg6vw_oa9HCdzENAB6w262'  # replace with a valid bingmap api key
base_url = "https://dev.virtualearth.net/REST/v1/Imagery/Map/"
zoom = "20"
maptype = "Aerial"
size = "600,600"

# latlon = '53.94922927940894, -1.0388568943295285'

i =0
j = 0 
k = 0
for i in range(20): 
	lat = 53.934605 + (i/1000)
	for j in range(20):
		lon = -1.139376 + (j/1000)
		latlon = str(lat)+", "+str(lon)
		print(latlon)
		url = base_url + maptype + '/' + latlon + '/' + zoom + '?mapsize=' + size + "&fmt=png&key=" + api_key
		print(url)
		response = requests.get(url)
		with open('/Users/Hadley 1/Documents/MEng Project/python/bing_images/image'+str(k)+'.png', 'wb') as file:   # save the image to .png file
			file.write(response.content)
		k +=1
	j =0
import requests
import urllib
import simplejson
import json

api_key = 'AjH2UXENFQRLK2RYNaLh5yMK6lbVNIt9RVUkssIzONIg6vw_oa9HCdzENAB6w262'  # replace with a valid bingmap api key
base_url = "https://dev.virtualearth.net/REST/v1/Imagery/Map/"
base_url_postcode = "http://dev.virtualearth.net/REST/v1/Locations/GB/"
zoom = "20"
maptype = "Aerial"
size = "600,600"


def bing_getaerial(latlon):
    url = base_url + maptype + '/' + latlon + '/' + zoom + '?mapsize=' + size + "&fmt=png&key=" + api_key
    response = requests.get(url)
    with open('./mapsamples/testmap.png', 'wb') as file:   # save the image to .png file
        file.write(response.content)
    return



def find_postcode(postcode):
    url = base_url_postcode + postcode +'?key=' + api_key
    response = requests.get(url)
    json_object = json.loads(response.text)
    latlong = str(json_object['resourceSets'][0]['resources'][0]['point']['coordinates'][0])+", "+str(json_object['resourceSets'][0]['resources'][0]['point']['coordinates'][1])        
    print(latlong)
    return latlong 

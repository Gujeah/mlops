
import requests

ride={ 
    "PULocationID":223,
    "DOLocationID":22,
   "trip_distance":12
}

url='http://localhost:9696/predict'
response=requests.post(url, json=ride)
print (response.json())

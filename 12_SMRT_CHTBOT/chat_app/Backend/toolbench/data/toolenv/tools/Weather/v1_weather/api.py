import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


#https://rapidapi.com/apininjas/api/weather-by-api-ninjas/
def v1_weather(country: str=None, state: str=None, city: str='Seattle'):
	

	url = "https://weather-by-api-ninjas.p.rapidapi.com/v1/weather"

	#querystring = {"city":city,"country":country}
	querystring = {}
	if country:
		querystring['country'] = country
	if state:
		querystring['state'] = state
	if city:
		querystring['city'] = city


	headers = {
		"X-RapidAPI-Key": "f5ce08b98fmsh493ce5f935f768fp113ce0jsn156ad2cdf9db",
		"X-RapidAPI-Host": "weather-by-api-ninjas.p.rapidapi.com"
	}

	response = requests.get(url, headers=headers, params=querystring)

	try:
		observation = response.json()
	except:
		observation = response.text
	return observation
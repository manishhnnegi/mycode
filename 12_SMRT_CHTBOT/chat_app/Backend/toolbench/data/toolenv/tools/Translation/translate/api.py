import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def accurate_language_detection(text: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return the Language of the Text"
    
    """
    url = f"https://translate287.p.rapidapi.com/detect/accurate"
    querystring = {'text': text, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "translate287.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fast_language_detection(text: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return the Language of the Text"
    
    """
    url = f"https://translate287.p.rapidapi.com/detect/fast"
    querystring = {'text': text, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "translate287.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def translate(text: str, dest: str, src: str='auto', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return Translated Text and the source language if it wasn't specified"
    
    """
    url = f"https://translate287.p.rapidapi.com/translate/"
    querystring = {'text': text, 'dest': dest, }
    if src:
        querystring['src'] = src
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "translate287.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

import requests

def language_translator(source_language: str, target_language: str,text: str):
    """
    "An Endpoint to fetch Arrivals on a given date"

    """

    url = "https://text-translator2.p.rapidapi.com/translate"

    payload = {
        "source_language": source_language,
        "target_language": target_language,
        "text": text
    }
    headers = {
        "content-type": "application/x-www-form-urlencoded",
        "X-RapidAPI-Key": "f5ce08b98fmsh493ce5f935f768fp113ce0jsn156ad2cdf9db",
        "X-RapidAPI-Host": "text-translator2.p.rapidapi.com"
    }

    response = requests.post(url, data=payload, headers=headers)

    try:
        observation = response.json()
    except:
        observation = response.text

    return observation
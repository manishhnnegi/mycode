import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search(query: str, brand: str=None, min_price: int=None, max_price: int=None, country: str='IN', category_id: str='aps', sort_by: str=None, page: str='1'):
    """
    "Search for product offers on Amazon with support for multiple filters and options."
    query: Search query. Supports both free-form text queries or a product asin.
        brand: Find products with a specific brand. Multiple brands can be specified as a comma (,) separated list. The brand values can be seen from Amazon's search left filters panel, as seen [here](https://www.amazon.com/s?k=phone).

**e.g.** `SAMSUNG`
**e.g.** `Google,Apple`
        min_price: Only return product offers with price greater than a certain value. Specified in the currency of the selected country. For example, in case country=US, a value of *105.34* means *$105.34*.
        max_price: Only return product offers with price lower than a certain value. Specified in the currency of the selected country. For example, in case country=US, a value of *105.34* means *$105.34*.
        country: Sets the marketplace country, language and currency. 

**Default:** `US`

**Allowed values:**  `US, AU, BR, CA, CN, FR, DE, IN, IT, MX, NL, SG, ES, TR, AE, GB, JP`

        category_id: Find products in a specific category / department. Use the **Product Category List** endpoint to get a list of valid categories and their ids for the country specified in the request.

**Default:** `aps` (All Departments)
        sort_by: Return the results in a specific sort order.

**Default:** `RELEVANCE`

**Allowed values:** `RELEVANCE, LOWEST_PRICE, HIGHEST_PRICE, REVIEWS, NEWEST`

        page: Results page to return.

**Default:** `1`
        
    """
    url = f"https://real-time-amazon-data.p.rapidapi.com/search"
    querystring = {'query': query, }
    if brand:
        querystring['brand'] = brand
    if min_price:
        querystring['min_price'] = min_price
    if max_price:
        querystring['max_price'] = max_price
    if country:
        querystring['country'] = country
    if category_id:
        querystring['category_id'] = category_id
    if sort_by:
        querystring['sort_by'] = sort_by
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": "f5ce08b98fmsh493ce5f935f768fp113ce0jsn156ad2cdf9db",
            "X-RapidAPI-Host": "real-time-amazon-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text

    observation = observation['data']['products'][:4]
    return observation


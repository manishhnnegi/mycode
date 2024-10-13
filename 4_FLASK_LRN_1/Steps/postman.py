import requests
import json

# Base URL
base_url = "http://127.0.0.1:5000"

# 1. Create Store
store_data = {
    "name": "My Store"
}
response = requests.post(f"{base_url}/store", json=store_data)
print("Create Store:", response.json())



# 2. Get Store
store_id = response.json()['id']
response = requests.get(f"{base_url}/store/{store_id}")
print("Get Store:", response.json())



# 3. Create Item
item_data = {
    "name": "Chair",
    "price": 15.88,
    "store_id": store_id
}
response = requests.post(f"{base_url}/item", json=item_data)
print("Create Item:", response.json())

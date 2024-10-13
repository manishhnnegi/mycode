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
store_id = "3fb652a788254b39b9b29e5c22675f74"
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

# 4. Get Item
item_id = "ab3eb2cf09e04581b30de628371a1eac"
response = requests.get(f"{base_url}/item/{item_id}")
print("Get Item:", response.json())

# 5. Create Another Store
store_data2 = {
    "name": "My father's Store"
}
response = requests.post(f"{base_url}/store", json=store_data2)
print("Create Store 2:", response.json())

# 6. Get Another Store
store_id2 = "8a5f99c0cdec45c58cdf799e56465817"
response = requests.get(f"{base_url}/store/{store_id2}")
print("Get Store 2:", response.json())

# 7. Create Another Item
item_data2 = {
    "name": "Chair",
    "price": 30.88,
    "store_id": "b388bfa0263c4f7a988957a8c01ce6d9"
}
response = requests.post(f"{base_url}/item", json=item_data2)
print("Create Item 2:", response.json())

# 8. Create Another Item in Store 1
item_data3 = {
    "name": "Table",
    "price": 10.88,
    "store_id": store_id
}
response = requests.post(f"{base_url}/item", json=item_data3)
print("Create Item 1:", response.json())

# 9. Create Another Item in Store 2
item_data4 = {
    "name": "Table",
    "price": 20.88,
    "store_id": "b388bfa0263c4f7a988957a8c01ce6d9"
}
response = requests.post(f"{base_url}/item", json=item_data4)
print("Create Item 2.1:", response.json())

# 10. Get All Stores and Items Data
response = requests.get(f"{base_url}/all_stores")
print("All Store and Item Data:", response.json())

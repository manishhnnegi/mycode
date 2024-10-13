import uuid
from flask import Flask, request, abort
from db import stores, items

app = Flask(__name__)

##################################### store ######################################
@app.get("/store/<string:store_id>")
def get_store(store_id):

    try:
        return stores[store_id]
    except ValueError:
        abort(404, message= "Store not Found")
    

@app.post("/store")
def create_store():
    store_data = request.get_json()    #store_data = {'name': 'My Store'}
    if "name" not in store_data:
        abort(400, message = "Bad request. Ensure 'name' is included n the JSON payload.",
        )
    for store in stores.values():  # stores = {"store_id1":{'name': 'My Store', 'id': 'store_id1'}}
        if store_data["name"] == store["name"]:
            abort(400, message = f"store is already exists")
        
    store_id = uuid.uuid4().hex
    store = {**store_data, "id": store_id}
    stores[store_id] = store

    return store
    
@app.delete("/store/<string:store_id>")
def delete_store(store_id):
    try:
        del stores[store_id]
        return {"message":"Store deleted"}
    except KeyError:
        abort(404, message = "Store not found")



@app.get("/store")
def get_stores():
    return {"stores" : list(stores.values())}




########################################## items ###################################
@app.get("/item/<string:item_id>")
def get_item(item_id):
    try:
        return items[item_id]
    except ValueError:
        abort(404, message="Item not Found")

@app.post("/item")
def create_item():
    item_data = request.get_json()  #item_data = {"name":"Chair","price":15.88, "store_id": "dgghdfhd123"}
    if ('price' not in item_data
        or 'store_id' not in item_data
        or 'name' not in item_data):
        abort(404, message= "Bad request. Ensure the 'price','name', and 'store_id' are included in the JSON payload.")

    for item in  items.values():  #items={"store_id1":{"item_id": "item_id1","name":"Chair","price":15.88}}
        if (item_data['store_id']== item['store_id']
            and item_data['name'] == item['name']
            ):
            abort(400, message= f"Item already exists.")

    item_id = uuid.uuid4().hex
    item = {**item_data, "id": item_id}
    items[item_id] = item

    return item
    

@app.delete("/item/<string:item_id>")
def delete_item(item_id):
    try:
        del items[item_id]
        return {"message": "Item deleted"}
    except KeyError:
        abort(404, message= "Item not Found")


@app.put("/item/<string:item_id>")
def update_item(item_id):
    item_data = request.get_json()
    if ('price' not in item_data
        or 'name' not in item_data):
        abort(404, message= "Bad request. Ensure the 'price','name', and 'store_id' are included in the JSON payload.")
    try:
        item = items[item_id]
        item = item | item_data
        items[item_id] = item
        return item
    except KeyError:
        abort(404, message= "Item not found.")


@app.get("/item")
def get_all_items():
    return {"items": list(items.values())}

#######################################################
@app.get("/all_stores")
def all_store_details():
    return {"stores:": stores,  "items:":items}



if __name__ == "__main__":
    app.run(debug= True)



















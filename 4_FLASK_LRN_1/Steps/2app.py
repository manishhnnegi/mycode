from flask import Flask, request,abort
from db import items, stores

import uuid

app = Flask(__name__)


@app.get("/store")      # this is api end point-----> is /store and the fun associated withn it is get_store
#@app.route("/store", methods=["GET"])
def get_stores():       # it will get accessed by--->  http://127.0.0.2.5000/store--- its an addersss to acces the end point
    return {"stores": list(stores.values())}


@app.post("/store")
def create_store():
    #pass
    store_data = request.get_json()   # convert json to dictonary which is coming from the server
    store_id = uuid.uuid4().hex
    store = {**store_data, "id" : store_id}
    stores[store_id] = store
   
    return store, 201


@app.post("/item")
def create_item():
    # print("here---------------.",name)
    # pass
    item_data = request.get_json()

    if item_data["store_id"] not in stores:
        #return {"message": "store not found"}, 404  
        abort(404, message= "store nt found")
    
    item_id = uuid.uuid4().hex
    item = {**item_data, "id":item_id}
    items[item_id] = item
    
    return item, 201
    
@app.get("/item")
def get_all_items():
    return {"items": list(items.values())}
    


@app.get("/store/<string:store_id>")
def get_store(store_id):
    try:
        return stores[store_id]    
    except KeyError:  
        #return {"message": "Store not Found"}, 404
        abort(404, message= "store not found")


@app.get("/item/<string:item_id>")
def get_item(item_id):
    try:
        return items[item_id]
    except KeyError:
        #return {'message': "stroe not found"}, 404
        abort(404, message= "key not found" )




if __name__ == "__main__":
    app.run(debug=True)
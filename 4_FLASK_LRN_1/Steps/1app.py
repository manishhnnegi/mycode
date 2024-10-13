from flask import Flask, request

app = Flask(__name__)

stores = [
    {
        "name": "My Store",
        "items": [
            {"name":"Chair",
             "price":15.88
             }
        ]
    }
]


@app.get("/store")      # this is api end point-----> is /store and the fun associated withn it is get_store
#@app.route("/store", methods=["GET"])
def get_stores():       # it will get accessed by--->  http://127.0.0.2.5000/store--- its an addersss to acces the end point
    return {"stores": stores}


@app.post("/store")
def create_store():
    #pass
    request_data = request.get_json()   # convert json to dictonary which is coming from the server
    new_store = {"name":request_data['name'], "item":[]}
    stores.append(new_store)
    return new_store, 201


@app.post("/store/<string:name>/item")
def create_item(name):
    # print("here---------------.",name)
    # pass
    request_data = request.get_json()
    for store in stores:
        if store['name'] == name:
            new_item = {"name": request_data['name'], "price": request_data['price']}
            store['items'].append(new_item)
            return new_item, 201
    
    return {"message": "store not found"}, 404


@app.get("/store/<string:name>")
def get_store(name):
    for store in stores:
        if store['name']== name:
            return store    # its good to retrun json istead of list but you can also retrun list as well 
    return {"message": "Store not Found"}, 404


@app.get("/store/<string:name>/item")
def get_item_in_store(name):
    for store in stores:
        if store['name']==name:
            return {"items": store['items']}
    return {'message': "stroe not found"}, 404




if __name__ == "__main__":
    app.run(debug=True)
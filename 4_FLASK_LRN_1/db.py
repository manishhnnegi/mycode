stores = {}
items = {}       

#### structure of items

#items = {
#           "store_id1":{"item_id": "item_id1","name":"Chair","price":15.88},
#           "store_id2":{"item_id": "item_id2","name":"Table","price":13.88}
#          }

#### structure of stores

# {
#     "store_id1":{'name': 'My Store', 'id': 'store_id1'},
#      "store_id2":{'name': 'My Store', 'id': 'store_id2'},
#     }



#  user will send following info:

##   item_data = {"name":"Chair","price":15.88, "store_id": "dgghdfhd123"}

##   store_data = {'name': 'My Store'}

## for update    item_data =  {"name":"Table","price":200.88} and    key in url
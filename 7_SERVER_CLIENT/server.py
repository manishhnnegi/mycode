from flask import Flask, request, jsonify

app = Flask(__name__)
data_store = {
    1: {"name": "Mahabharata", "author": "ved vyasa"},
    2: {"name": "Ramayana", "author": "Valamiki"}
}

@app.route('/')
def home():
    return "Wellcome to the Book store API!"

@app.route('/books', methods = ['GET'])
def get_books():
    return jsonify(data_store)

@app.route('/books', methods=['POST'])
def add_book():
    new_book = request.json
    print(new_book)
    book_id = len(data_store) + 1
    data_store[book_id] = new_book
    return jsonify({"message": "Book added", "book_id": book_id}), 201


# DELETE request to delete a book
@app.route('/books/<int:book_id>', methods=['DELETE'])
def delete_book(book_id):
    if book_id in data_store:
        del data_store[book_id]
        return jsonify({"message": "Book deleted"}), 200
    else:
        return jsonify({"error": "Book not found"}), 404
    

# PUT request to update an existing book
@app.route('/books/<int:book_id>', methods=['PUT'])
def update_book(book_id):
    if book_id in data_store:
        updated_data = request.json
        data_store[book_id].update(updated_data)
        return jsonify({"message": "Book updated"}), 200
    else:
        return jsonify({"error": "Book not found"}), 404

if __name__ == '__main__':
    app.run(debug=True)
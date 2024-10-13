from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Sample in-memory data store
books = {
    1: {"name": "Mahabharata", "author": "ved vyasa"},
    2: {"name": "Ramayana", "author": "Valamiki"}
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/books', methods=['GET'])
def get_books():
    return jsonify(books)

@app.route('/books', methods=['POST'])
def add_book():
    data = request.get_json()
    new_id = max(books.keys()) + 1
    books[new_id] = data
    return jsonify({"message": "Book added successfully!"})

@app.route('/books/<int:book_id>', methods=['PUT'])
def update_book(book_id):
    data = request.get_json()
    if book_id in books:
        books[book_id].update(data)
        return jsonify({"message": "Book updated successfully!"})
    return jsonify({"message": "Book not found!"}), 404

@app.route('/books/<int:book_id>', methods=['DELETE'])
def delete_book(book_id):
    if book_id in books:
        del books[book_id]
        return jsonify({"message": "Book deleted successfully!"})
    return jsonify({"message": "Book not found!"}), 404

if __name__ == '__main__':
    app.run(debug=True)

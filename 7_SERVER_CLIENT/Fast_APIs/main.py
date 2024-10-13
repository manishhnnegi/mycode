from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict

app = FastAPI()

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# In-memory storage for books
books: Dict[int, Dict[str, str]] = {}
next_id = 1

class Book(BaseModel):
    name: str
    author: str

@app.post("/books", status_code=201)
async def add_book(book: Book):
    global next_id
    book_id = next_id
    next_id += 1
    books[book_id] = book.dict()
    return {"message": "Book added", "id": book_id}

@app.put("/books/{book_id}")
async def update_book(book_id: int, book: Book):
    if book_id not in books:
        raise HTTPException(status_code=404, detail="Book not found")
    books[book_id] = book.dict()
    return {"message": "Book updated"}

@app.delete("/books/{book_id}")
async def delete_book(book_id: int):
    if book_id not in books:
        raise HTTPException(status_code=404, detail="Book not found")
    del books[book_id]
    return {"message": "Book deleted"}

@app.get("/books")
async def get_books():
    return books

@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("templates/index.html") as f:
        return f.read()

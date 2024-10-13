from fastapi import FastAPI, Body
from pydantic import BaseModel, Field
from typing import Optional

app = FastAPI()


class Book:
    id:int
    title:str
    author: str
    description:str
    rating:str
    published_date:str


    def __init__(self,id, title, author, description, rating, published_date ):
        self.id = id
        self.title = title
        self.author = author
        self.description = description
        self.rating = rating
        self.published_date = published_date


class BookRequest(BaseModel):
    id: Optional[int] = Field(description='ID is not needed on creation', default= None)
    title:str = Field(min_length= 3)
    author:str = Field(min_length=1)
    description :str = Field(min_length= 1, max_length= 100)
    rating:int = Field(gt=0,lt=6)
    published_date:int = Field(gt=1999, lt=2031)

    model_config = {
        "json_schema_extra": {
            "example": {
                "title": "A new book",
                "author": "codingwithroby",
                "description": "A new description of a book",
                "rating": 5,
                'published_date': 2029
            }
        }
    }



BOOKS = [
    Book(1, "computer Science Pro", "codingwithroby", "A very nice book!", 5, 2030)
    
    ]

@app.get("/books")
async def read_all_books():
    return BOOKS


@app.post("/create-book")
async def create_book(book_request: BookRequest):
    new_book = Book(**book_request.model_dump())
    return BOOKS.append(new_book)

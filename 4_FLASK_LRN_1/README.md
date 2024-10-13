# course outline

1. about flask 

    There are two parts to a Flask route:

    The endpoint decorator
    The function that should run
    The endpoint decorator (@app.get("/store")) registers the route's endpoint with Flask. That's the /store bit. That way, the Flask app knows that when it receives a request for /store, it should run the function.

    The function's job is to do everything that it should, and at the end return something. In most REST APIs, we return JSON, but you can return anything that can be represented as text (e.g. XML, HTML, YAML, plain text, or almost anything else).

2. about json
    json is the string of dict or list(imp)

3. errors
    <!doctype html>
    <html lang=en>
    <title>405 Method Not Allowed</title>
    <h1>Method Not Allowed</h1>
    <p>The method is not allowed for the requested URL.</p>


    500 internal server error when defined function not able to send the desired value.
    201 means hv accepted the data in the backend  retrun status code

4. source notes
    lecture notes:
    https://rest-apis-flask.teclado.com/docs/flask_smorest/improvements_on_first_rest_api/
    
    code repo:
    https://github.com/tecladocode/rest-apis-flask-python#getting-started
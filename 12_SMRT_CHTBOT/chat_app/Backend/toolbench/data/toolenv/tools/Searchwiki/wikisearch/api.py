from langchain_community.utilities.wikipedia import WikipediaAPIWrapper


#https://rapidapi.com/apininjas/api/weather-by-api-ninjas/
def wikisearch(query: str=None):
    dd = WikipediaAPIWrapper()
    return dd.run(query)
	
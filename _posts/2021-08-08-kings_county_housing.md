# Calculating the Most Lucrative Renovations Based on Kings County Housing Dataset


![describe_data](https://user-images.githubusercontent.com/71570329/128657233-cd2279ea-54d3-4d8c-a947-e1bf6e36bc03.png)

My second data science project with Flatiron School is an analysis of housing prices from the  [Kings County House Sales Dataset](https://www.kaggle.com/harlfoxem/housesalesprediction). 

Feel free to fork my [project repo](https://github.com/clairesarraille/ph2finproj) if you'd like to follow along and examine the code more closely.

## Retrieving Data from the TMDb API

I decided to use The Movie Database (TMDb) data and a nifty python library to handle requests and responses called [tmdbv3api](https://github.com/AnthonyBloomer/tmdbv3api). Using its discover.discover_movies() and movies.details() methods, I was able to extract data two ways: first by getting data per year, sorted by descending revenue, and then by passing that result to movies.details() to retrieve the information I wanted to visualize. Here's the basic function I used to retrieve data from discover.discover_movies():

```python

movies = {}

def return_discover_movies():
    for year in N_YEAR_RANGE:
        movies[year] = []
        for page in range(0, N_PAGES):
            movies_running = discover.discover_movies({
                'sort_by': 'revenue.desc',
                'primary_release_year': str(year),
                'page': str(page + 1),
                'vote_count.gte': 1,
                'with_original_language': 'en'
            })
            movies[year].extend([dict(tmdb_obj)
                                for tmdb_obj in movies_running])

```

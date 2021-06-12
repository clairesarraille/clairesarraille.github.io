## My first data science project: Analyzing Movie Success using TMDb Data

My first data science with Flatiron School is an analysis of movie success in terms of net revenue. Imagining that I was helping develop early R&D for an upcoming movie studio, I looked at several variables that I hypothesized might impact revenue.

I decided to use TMDb data and a nifty python library to handle requests and responses called [tmdbv3api](http://google.com). Using its discover.discover_movies() and movies.details() methods, I was able to extract data two ways: first by getting data per year, sorted by descending revenue, and then by passing that result to movies.details() to retrieve the information I wanted to visualize. Here's the basic function I used to retrieve data from discover.discover_movies():

```python
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

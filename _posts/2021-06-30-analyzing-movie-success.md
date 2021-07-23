# Analyzing Movie Success using TMDb Data

![q3](https://user-images.githubusercontent.com/71570329/121889302-ed81c980-cccd-11eb-9350-b0a8f84acb32.png)


My first data science project with Flatiron School is an analysis of movie success in terms of net revenue (profit) from 2016-June 2021. Imagining that I was helping develop early R&D for an upcoming movie studio, I started with a collection of the 3,000 highest revenue movies of 2016-2021. I threw out those not originally shot in English, and any missing a genre or production company. I further narrowed the scope to the upper quartile of profit, resulting in a neat 606 titles to focus on for the majority of the visualizations.

Feel free to fork my [project repo](https://github.com/clairesarraille/mod1finproj) if you'd like to follow along and examine the code more closely. If you're a relative newbie to data science like I am, you'll likely enjoy the level of documentation and commenting in my Jupyter Notebook :)

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

Note: If you use Anaconda for python package management check out the documentation on hiding your API key [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#saving-environment-variables)

```
priv_api_key = os.environ.get('TMDB_PRIVATE_API_KEY')
tmdb.api_key = priv_api_key
```

With my 3,000 of the highest-revenue movies of 2016-2021, I did some very quick manual review to help spot patterns of error. I appended all titles and movie ids to a .txt file. There were a few I followed up on, but they all turned out to be legitimate after a little googling.

For example, I found a possible duplicate that turned out to be two separate movies:
```
Split
2016-04-07
The story of a young woman who takes an epic journey to claim her own darkness...

Split
2016-11-15
Though Kevin has evidenced 23 personalities to his trusted psychiatrist, Dr. Fletcher...
```

Then it was time to view my data in a pandas dataframe for a bit more initial exploratory analysis. The movies dictionary created with the  return_discover_movies() function above has keys for years 2016-2021. The value for each year is a list of dictionaries. Each dictionary is a movie. To turn this data into a Pandas datastructure, we iterate through all the years and all the dictionaries in each year. Our result, movies_list_of_dict is a simple list of all the movie dictionaries. All we have to do then is use pd.DataFrame() to use the keys as column headers and the values as the cell values in rows. Below is a screenshot of the list in the movies dictionary for 2021:

```python
movies_list_of_dict = [movie_d for movie_yr in movies for movie_d in movies[movie_yr]]
df_movies = pd.DataFrame(movies_list_of_dict)
```

![movies2021](https://user-images.githubusercontent.com/71570329/121879073-4b5be480-ccc1-11eb-8f71-c63691dbf763.png)



As you can see from the above screenshot, I didn't have revenue data yet. My next step was to retrieve the columns I needed via movie.details() method, which translates to this part of TMDB's API, [/movie/{movie_id}](https://developers.themoviedb.org/3/movies/get-movie-details). A way I dealt with any anticipated missing fields for each movie: In case any of the movies are missing any of the below keys, I used dict.get(), which returns None instead of throwing an error if a given key doesn't exist.

```python
my_keys = ['id', 'title', 'release_date', 'revenue', 'budget', 'genres', 'original_language',
           'spoken_languages', 'popularity', 'production_companies', 'production_countries', 'runtime', 'keywords']

details_list = []
details_list.extend({my_key: dict(movie.details(id_index)).get(my_key)
                    for my_key in my_keys} for id_index in movie_id_list)
                    
df_details = pd.DataFrame(details_list)
                    
```                  

## Data Cleaning and Munging:
Because my imaginary studio is still in R&D mode, I focused on movies originally shot in English as one way to narrow the scope. Having specified for only originally English language films upon intake, this next step helped find foreign language films that were erroneously coded as 'with_original_language' = 'en.'

```python
def detect_en(ascii_string):
    try:
        ascii_string.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True
 ```

### Exploring Genre

The genre field was a list of tmdbv3api objects: a dictionary-like object. First I converted each object to a dictionary, and from each dictionary, extracted the genre name's string.
Data Format Before:

```
<class 'list'>
<class 'tmdbv3api.as_obj.AsObj'>
[{'id': 12, 'name': 'Adventure'}, {'id': 28, 'name': 'Action'}, {'id': 878, 'name': 'Science Fiction'}]
```

```python
def convert_to_list_of_dict(val):
    # val parameter is used to transform each value of the pandas genres series
    val = [dict(x) for x in val]
    return val

df_details['genres_list_dict'] = df_details.loc[:,
                                                'genres'].apply(convert_to_list_of_dict)
```                                             

Almost identical code as above, except this line: ```val = [x.get('name') for x in val]```, and I was left with this beautiful, useable datastructure:

```
<class 'list'>
<class 'str'>
['Adventure', 'Action', 'Science Fiction']
```

Wherever any genre list was empty, I converted those values to NaN and then removed them from my dataframe - I wanted to drill-down to titles that had both a genre and production company, so that I could see what combinations of these attributes are most common in high-profit movies.

```python
df_details['genres_list_str'] = df_details['genres_list_str'].apply(
    lambda x: np.nan if len(x) == 0 else x)

df_details.dropna(subset = ["genres_list_str"], inplace=True)
```

I went through the exact same steps for production company and created a copy of my dataframe with only the columns I wanted to keep. After adding a profit column (revenue - budget), my data looked like this:

```
	title                               genre                                               production_company          profit
0	Captain America: Civil War          ['Adventure', 'Action', 'Science Fiction']          ['Marvel Studios']          $ 903,296,293
1	Rogue One: A Star Wars Story        ['Action', 'Adventure', 'Science Fiction']          ['Lucasfilm Ltd.']          $ 856,057,273

```

The final way I filtered my data was to isolate the upper quartile or 75th percentile for profit.

```
# I examined the results from .describe() to get the upper quartile for profit:
df[['revenue','budget','profit','popularity','runtime']].describe()
df_q3 = df[(df['revenue'] >= 9289990)].copy()
```

## Visualize Frequency and Co-occurrence of Genre

At this point, I was super excited to finally visualize my carefully cleaned & curated dataset :) However, there was still some serious data transformation necessary to deal with the list-of-string datastructure I had stored for genre and production company. At this point in my work, I learned invaluable techniques for dealing with list values in columns from Max Hilsdorf's wonderful and detailed tutorial on the subject: [Dealing with List Values in Pandas Dataframes](https://towardsdatascience.com/dealing-with-list-values-in-pandas-dataframes-a177e534f173). I was incredibly grateful to have his blog post and Jupyter Notebook tutorial as a reference. It's amazing how much you can learn from well-documented code! I had a lot of fun figuring out what he had done and finally feeling like I could "see" what was happening in the process of numpy's dot product operation. The main idea of the process is to take a list of strings, and convert it to a matrix of boolean values, where each column is one unique string inside the list. This involves first creating a giant list with all the instances of each string (so each string will repeat the number of times it appears in a field within the original dataframe. At that point, .value_counts() works on the big list of strings. For genre, these are the value counts I got:

```
Drama              246
Comedy             213
Action             192
Thriller           163
Adventure          156
Horror              92
Family              91
Fantasy             90
Science Fiction     89
Crime               89
Animation           65
Romance             59
Mystery             58
History             53
Music               24
War                 16
Documentary          6
Western              4
dtype: int64
```

Then, we take these categorical data and counts for each category, and use it to produce a dataframe of T/F values, where each column is one of the categories, and each row represents an individual movie!

My version of Max's elegant solution to creating a boolean mask from a Pandas Series containing categories and value counts:


```python

def make_boolean_mask(p_series, u_series):
    # Instantiate Boolean Dictionary:
    bool_dict = {}
    # Loop through pandas Series containing unique categories (such as genre or production company):
    for counter, u_value in enumerate(u_series):
        # Build bool_dict with keys that are each unique category, and values that are a boolean for each items in each row of a pandas Series, such as genre
        bool_dict[u_value] = p_series.apply(lambda x: u_value in x)
    # Then, convert this dictionary to a data frame:
    return pd.DataFrame(bool_dict)
```


```python
# Using .keys() on a Pandas Series returns the index labels
genre_boolean = make_boolean_mask(df_q3["genre"], unique_genres.keys()) 
```

From this dataframe of boolean values, the next step was to convert from T/F to 0/1. And finally, to create a matrix of all co-occurrences of our category (in this case, genre).

![mask](https://user-images.githubusercontent.com/71570329/121886606-73037a80-ccca-11eb-9e26-080f7e29109d.png)


This humble bit of code converts our boolean dataframe of 18 categories of genre to a 18 * 18 matrix. Each list in the matrix is one genre (Drama, Comedy, etc in that order). And each item in a list represents the number of times that genre co-occurs with another genre.

```
genre_matrix = np.dot(bin_genres.T, bin_genres)
```

For example, the first list in the matrix below represents the number of times Drama co-occurs with every other genre in the order of the columns: Drama with Drama, Drama with Comedy, Drama with Action, Drama with Thriller, and so on. Thus, it makes sense why the very first value in the first list is 246, because that's the total number of times Drama occurs. The very last value in the very last list is 4 (not shown), because that's the number of times Western occurs.

```
genre_matrix

array([[246,  59,  33,  62,  21,  15,  11,  16,  16,  40,   4,  43,  24,
         51,  14,  14,   0,   2],
       [ 59, 213,  46,  11,  60,  11,  65,  36,  13,  26,  54,  28,  10,
          6,  15,   2,   0,   0],
       [ 33,  46, 192,  64,  78,  13,  12,  33,  62,  44,  11,   3,   7,
         11,   0,   7,   0,   1],
(...)
```

At last, this matrix could be converted to a Pandas dataframe and visualized with a heatmap:

```
# Convert Matrix to DataFrame:
genre_frequency = pd.DataFrame(genre_matrix, columns = unique_genres.keys(), index = unique_genres.keys())
```

```
# Used seaborn to visualize the co-occurrence of genres for Q3 Movies:
fig, ax = plt.subplots(figsize = (13,7))
sn.heatmap(genre_frequency, cmap = "Greens").set(title='Genre Co-occurrence for Q3 Profit Movies on TMDb, 2016-2021')
plt.xticks(rotation=50)
```

![genre](https://user-images.githubusercontent.com/71570329/125541508-7e30b64e-0e02-4fb1-a5c6-024215207fd7.png)



I repeated the whole process of munging, boolean-masking, and vector multiplication for production company!

![prod_company](https://user-images.githubusercontent.com/71570329/125541478-4ab3b657-d0e1-40f7-88ae-204ee8b40e91.png)


### Conclusion
What I love about these heat maps is how information-dense they are while remaining easy to read and intuitive. As you can see, a lot of work goes into creating something seemingly so simple.
We can observe from the genre and production company heat maps that some of the highest-profit movies in the past 5 years are Dramas, Comedies and Action films produced by Universal, Warner Bros, and Columbia. Even more interesting are the winning combinations of these attributes. 20th Century Fox paired with Columbia, Sony paired with Warner Bros., and TSG with Lionsgate appear to be synergistic forces. Likewise, we can observe patterns of genre that appear to garner the greatest rewards at the box office, such as action-adventure, historical dramas, and family comedies.

I had an amazing time exploring these data and learned an incredible amount in a short period. Stay tuned for more project posts in the upcoming months.

This product uses the TMDb API but is not endorsed or certified by TMDb
![attribution](https://user-images.githubusercontent.com/71570329/121875052-ddadb980-ccbc-11eb-9f8d-0c3ef39b45fd.png)






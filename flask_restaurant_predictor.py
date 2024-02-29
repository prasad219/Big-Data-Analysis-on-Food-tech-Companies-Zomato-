from flask import Flask, render_template, request
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from nltk.corpus import stopwords

# Initialize Flask app
app = Flask(__name__)

# Load and preprocess data
zomato_real = pd.read_csv("newzomato.csv")

# Deleting Unnnecessary Columns
zomato = zomato_real.drop(['url', 'dish_liked', 'phone'], axis=1)

# Removing the Duplicates
zomato.drop_duplicates(inplace=True)

# Remove the NaN values from the dataset
zomato.dropna(how='any', inplace=True)

# Changing the column names
zomato = zomato.rename(columns={'approx_cost(for two people)': 'cost', 'listed_in(type)': 'type', 'listed_in(city)': 'city'})

# Some Transformations
zomato['cost'] = zomato['cost'].astype(str)
zomato['cost'] = zomato['cost'].apply(lambda x: x.replace(',', '.'))
zomato['cost'] = zomato['cost'].astype(float)

# Removing '/5' from Rates
zomato = zomato.loc[zomato.rate != 'NEW']
zomato = zomato.loc[zomato.rate != '-'].reset_index(drop=True)
remove_slash = lambda x: x.replace('/5', '') if type(x) == str else x
zomato.rate = zomato.rate.apply(remove_slash).str.strip().astype('float')

# Adjust the column names
zomato.name = zomato.name.apply(lambda x: x.title())
zomato.online_order.replace(('Yes', 'No'), (True, False), inplace=True)
zomato.book_table.replace(('Yes', 'No'), (True, False), inplace=True)

## Computing Mean Rating
restaurants = list(zomato['name'].unique())
zomato['Mean Rating'] = 0

for i in range(len(restaurants)):
    zomato['Mean Rating'][zomato['name'] == restaurants[i]] = zomato['rate'][zomato['name'] == restaurants[i]].mean()

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(1, 5))
zomato[['Mean Rating']] = scaler.fit_transform(zomato[['Mean Rating']]).round(2)

## Lower Casing
zomato["reviews_list"] = zomato["reviews_list"].str.lower()

## Removal of Puctuations
import string

PUNCT_TO_REMOVE = string.punctuation


def remove_punctuation(text):
    """custom function to remove the punctuation"""
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))


zomato["reviews_list"] = zomato["reviews_list"].apply(lambda text: remove_punctuation(text))

## Removal of Stopwords
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words('english'))


def remove_stopwords(text):
    """custom function to remove the stopwords"""
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])


zomato["reviews_list"] = zomato["reviews_list"].apply(lambda text: remove_stopwords(text))

## Removal of URLS


def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)


zomato["reviews_list"] = zomato["reviews_list"].apply(lambda text: remove_urls(text))

zomato[['reviews_list', 'cuisines']].sample(5)

# RESTAURANT NAMES:
restaurant_names = list(zomato['name'].unique())


def get_top_words(column, top_nu_of_words, nu_of_word):
    vec = CountVectorizer(ngram_range=nu_of_word, stop_words='english')
    bag_of_words = vec.fit_transform(column)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:top_nu_of_words]


zomato = zomato.drop(['address', 'rest_type', 'type', 'menu_item', 'votes'], axis=1)
import pandas as pd

# Randomly sample 60% of your dataframe
df_percent = zomato.sample(frac=0.5)

df_percent.reset_index(inplace=True)
df_percent.set_index('name', inplace=True)
indices = pd.Series(df_percent.index)

# Creating tf-idf matrix
tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=1, stop_words='english')

tfidf_matrix = tfidf.fit_transform(df_percent['reviews_list'])

cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

indices = pd.Series(df_percent.index)


# Define recommendation function
def recommend(name, cosine_similarities=cosine_similarities):
    # Create a list to put top restaurants
    recommend_restaurant = []

    # Find the index of the hotel entered
    idx = indices[indices == name].index[0]

    # Find the restaurants with a similar cosine-sim value and order them from biggest number
    score_series = pd.Series(cosine_similarities[idx]).sort_values(ascending=False)

    # Extract top 30 restaurant indexes with a similar cosine-sim value
    top30_indexes = list(score_series.iloc[0:31].index)

    # Names of the top 30 restaurants
    for each in top30_indexes:
        recommend_restaurant.append(list(df_percent.index)[each])

    # Creating the new data set to show similar restaurants
    df_new = pd.DataFrame(columns=['cuisines', 'Mean Rating', 'cost'])

    # Create the top 30 similar restaurants with some of their columns
    for each in recommend_restaurant:
        df_new = pd.concat([df_new, df_percent[['cuisines', 'Mean Rating', 'cost']][df_percent.index == each].sample()])

    # Drop the same named restaurants and sort only the top 10 by the highest rating
    df_new = df_new.drop_duplicates(subset=['cuisines', 'Mean Rating', 'cost'], keep=False)
    df_new = df_new.sort_values(by='Mean Rating', ascending=False).head(10)

    return df_new


# Flask routes
@app.route('/')
def man():
    return render_template('home.html')
    
    
@app.route('/iindex')
def iindex():
    return render_template('iindex.html')
    
@app.route('/avgcost')
def avgcost():
    return render_template('avgcost.html')

@app.route('/photoperrating')
def photoperrating():
    return render_template('photoperrating.html')
    
@app.route('/restrodelivery')
def restrodelivery():
    return render_template('restrodelivery.html')
    
@app.route('/restrostate')
def restrostate():
    return render_template('restrostate.html')
    
@app.route('/typeofrestro')
def typeofrestro():
    return render_template('typeofrestro.html')  



@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        restaurant_name = request.form['restaurant_name']
        recommended_restaurants = recommend(restaurant_name)
        return render_template('result.html', recommended_restaurants=recommended_restaurants)


if __name__ == '__main__':
    app.run(debug=True)

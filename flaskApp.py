from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)

# Read data
ratings_file = 'https://raw.githubusercontent.com/bsoifer/PredictivoAvanzado/main/datos/rating_final.csv'
places_file = 'https://raw.githubusercontent.com/bsoifer/PredictivoAvanzado/main/datos/dataLocal.csv'
users_file = 'https://raw.githubusercontent.com/bsoifer/PredictivoAvanzado/main/datos/dataUser.csv'

df_ratings = pd.read_csv(ratings_file)
df_places = pd.read_csv(places_file, sep=';')

if os.path.exists(users_file):
    df_users = pd.read_csv(users_file)
else:
    df_users = pd.DataFrame(columns=[
        'userID', 'latitude', 'longitude', 'smoker', 'drink_level', 'dress_preference', 
        'ambience', 'transport', 'marital_status', 'hijos', 'birth_year', 'interest', 
        'personality', 'religion', 'activity', 'weight', 'budget', 'height', 
        'Rcuisine', 'Upayment', 'rating', 'food_rating', 'service_rating'
    ])

df_users_filtered = df_users[df_users['userID'].str.startswith('U1')]

# Preprocessing
numerical_features = ['latitude', 'longitude', 'birth_year', 'weight', 'height']
categorical_features = ['smoker', 'drink_level', 'dress_preference', 'ambience', 'transport', 
                        'marital_status', 'hijos', 'interest', 'personality', 'religion', 
                        'activity', 'budget', 'Rcuisine', 'Upayment']

imputer = SimpleImputer(strategy='mean')
scaler = StandardScaler()
onehot = OneHotEncoder(handle_unknown='ignore', sparse=False)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[('imputer', imputer), ('scaler', scaler)]), numerical_features),
        ('cat', onehot, categorical_features)
    ])

user_features_matrix = preprocessor.fit_transform(df_users_filtered.drop(columns=['userID']))

user_similarity = cosine_similarity(user_features_matrix)

user_similarity_df = pd.DataFrame(user_similarity, index=df_users_filtered['userID'], columns=df_users_filtered['userID'])

def get_most_similar_user(user_data):
    user_data_df = pd.DataFrame([user_data])
    user_data_transformed = preprocessor.transform(user_data_df.drop(columns=['userID']))

    similarity_scores = cosine_similarity(user_data_transformed, user_features_matrix)
    most_similar_user_idx = similarity_scores.argsort()[0][-2]  # Use -2 to avoid self similarity
    most_similar_user_id = df_users_filtered.iloc[most_similar_user_idx]['userID']
    return most_similar_user_id

def get_user_top_recommendations(user_id):
    user_ratings = df_ratings[df_ratings['userID'] == user_id]
    sorted_ratings = user_ratings.sort_values(by='rating', ascending=False)

    recommendations = []
    for _, row in sorted_ratings.iterrows():
        place_id = row['placeID']
        place_details = df_places[df_places['placeID'] == place_id].iloc[0]
        recommendations.append({
            'name': place_details['name'],
            'address': place_details['address'],
            'rating': row['rating']
        })
    
    return recommendations

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_data = {
        'userID': 'U{}'.format(len(df_users) + 1),
        'latitude': request.form.get('latitude'),
        'longitude': request.form.get('longitude'),
        'smoker': request.form.get('smoker'),
        'drink_level': request.form.get('drink_level'),
        'dress_preference': request.form.get('dress_preference'),
        'ambience': request.form.get('ambience'),
        'transport': request.form.get('transport'),
        'marital_status': request.form.get('marital_status'),
        'hijos': request.form.get('hijos'),
        'birth_year': request.form.get('birth_year'),
        'interest': request.form.get('interest'),
        'personality': request.form.get('personality'),
        'religion': request.form.get('religion'),
        'activity': request.form.get('activity'),
        'weight': request.form.get('weight'),
        'budget': request.form.get('budget'),
        'height': request.form.get('height'),
        'Rcuisine': request.form.get('Rcuisine'),
        'Upayment': request.form.get('Upayment')
    }

    for field in user_data:
        if not user_data[field]:
            user_data[field] = None

    most_similar_user = get_most_similar_user(user_data)
    recommendations = get_user_top_recommendations(most_similar_user)

    return render_template('recommendations.html', recommendations=recommendations)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=True)

from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split
import os
import uuid

app = Flask(__name__)

# Variables globales para los datos y el modelo
df_ratings = None
df_places = None
df_users = None
algo = None
reader = Reader(rating_scale=(0, 2))  # Define reader globalmente

# Cargar los datos inicialmente
def load_data():
    global df_ratings, df_places, df_users
    # Obtener la ruta del directorio actual del script
    current_dir = os.path.dirname(__file__)

    # Rutas a los archivos CSV
    ratings_file = os.path.join(current_dir, 'datos', 'rating_final.csv')
    places_file = os.path.join(current_dir, 'datos', 'dataLocal.csv')
    users_file = os.path.join(current_dir, 'datos', 'dataUser.csv')

    # Cargar los DataFrames desde los archivos CSV
    df_ratings = pd.read_csv(ratings_file, sep=',')
    df_places = pd.read_csv(places_file, sep=';')
    df_users = pd.read_csv(users_file, sep=';')

    return Dataset.load_from_df(df_ratings[['userID', 'placeID', 'rating']], reader)

# Iniciar el modelo con los datos cargados
def init_model(data):
    global algo
    trainset, _ = train_test_split(data, test_size=0.2, random_state=42)
    algo = KNNBasic(k=20, sim_options={'name': 'msd', 'user_based': False})
    algo.fit(trainset)

# Preprocessing setup
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

def preprocess_users(df_users):
    return preprocessor.fit_transform(df_users.drop(columns=['userID']))

def get_most_similar_user(user_data, df_users, user_features_matrix):
    user_data_df = pd.DataFrame([user_data])
    user_data_transformed = preprocessor.transform(user_data_df.drop(columns=['userID']))

    similarity_scores = cosine_similarity(user_data_transformed, user_features_matrix)
    most_similar_user_idx = similarity_scores.argsort()[0][-2]  # Use -2 to avoid self similarity
    most_similar_user_id = df_users.iloc[most_similar_user_idx]['userID']
    return most_similar_user_id

def get_user_top_recommendations(user_id, df_ratings, df_places):
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

# Función para recomendar lugares
def recommend_places(user_id, algo, df_places, top_n=10):
    recommendations = []

    place_ids = df_places['placeID'].unique()

    for place_id in place_ids:
        prediction = algo.predict(user_id, place_id)
        recommendations.append((place_id, prediction.est))

    recommendations.sort(key=lambda x: x[1], reverse=True)

    top_recommendations = recommendations[:top_n]

    top_recommendations_details = []
    for place_id, rating in top_recommendations:
        place_info = df_places[df_places['placeID'] == place_id].iloc[0]
        top_recommendations_details.append({
            'Nombre': place_info['name'],  
            'Dirección': place_info['address'],  
            'Calificación Estimada': rating
        })

    return top_recommendations_details

# Ruta para recibir datos de usuario y generar recomendaciones
@app.route('/recommend', methods=['POST'])
def recommend_endpoint():
    user_data = request.json

    # Actualizar df_users con el nuevo usuario
    global df_users
    user_id = f"U20{uuid.uuid4().hex[:2].upper()}"
    new_user = pd.DataFrame([{'userID': user_id, 'latitude': None, 'longitude': None, **user_data}])
    df_users = pd.concat([df_users, new_user], ignore_index=True)
    df_users.to_csv(os.path.join(os.path.dirname(__file__), 'datos', 'dataUser.csv'), index=False)

    # Preprocess users and update similarity matrix
    user_features_matrix = preprocess_users(df_users)

    most_similar_user = get_most_similar_user(new_user.iloc[0], df_users, user_features_matrix)
    recommendations = get_user_top_recommendations(most_similar_user, df_ratings, df_places)

    return render_template('recommendations.html', recommendations=recommendations)

# Ruta para la página inicial
@app.route('/')
def index():
    unique_budget = ['medium', 'low', '?', 'high']
    unique_interest = ['variety', 'technology', 'none', 'retro', 'eco-friendly']
    unique_personality = ['thrifty-protector', 'hunter-ostentatious', 'hard-worker', 'conformist']
    unique_religion = ['none', 'Catholic', 'Christian', 'Mormon', 'Jewish']
    unique_activity = ['student', 'professional', '?', 'unemployed', 'working-class']
    unique_smoker = ['false', 'true', '?']
    unique_drink_level = ['abstemious', 'social drinker', 'casual drinker']
    unique_dress_preference = ['informal', 'formal', 'no preference', '?', 'elegant']
    unique_ambience = ['family', 'friends', 'solitary', '?']
    unique_transport = ['on foot', 'public', 'car owner', '?']
    unique_marital_status = ['single', 'married', 'widow', '?']
    unique_hijos = ['independent', 'kids', '?', 'dependent']
    unique_Rcuisine = ['American', 'Mexican', 'Bakery', 'Family', 'Cafe-Coffee_Shop', 'Diner',
                       'Latin_American', 'Japanese', 'Chinese', 'Pizzeria', 'Afghan', 'Regional',
                       'Contemporary', 'Middle_Eastern', 'Bar', 'Breakfast-Brunch', 'Cuban',
                       'Burgers', 'Sushi', 'Italian', 'Tex-Mex', 'Game', 'Cafeteria', 'Barbecue',
                       'Turkish', 'Organic-Healthy']
    unique_Upayment = ['cash', 'MasterCard-Eurocard', 'bank_debit_cards', 'VISA', 'American_Express']

    return render_template('index.html', 
                           unique_budget=unique_budget,
                           unique_interest=unique_interest,
                           unique_personality=unique_personality,
                           unique_religion=unique_religion,
                           unique_activity=unique_activity,
                           unique_smoker=unique_smoker,
                           unique_drink_level=unique_drink_level,
                           unique_dress_preference=unique_dress_preference,
                           unique_ambience=unique_ambience,
                           unique_transport=unique_transport,
                           unique_marital_status=unique_marital_status,
                           unique_hijos=unique_hijos,
                           unique_Rcuisine=unique_Rcuisine,
                           unique_Upayment=unique_Upayment)

if __name__ == '__main__':
    # Cargar los datos y el modelo al inicio
    data = load_data()
    init_model(data)

    # Obtener el puerto del entorno o usar el puerto 8080 por defecto
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=True)

import pandas as pd
import streamlit as st
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split

# Cargar datos
df_ratings = pd.read_csv('./datos/rating_final.csv')
df_places = pd.read_csv('./datos/geoplaces2.csv')

# Configurar Surprise
reader = Reader(rating_scale=(0, 2))
data = Dataset.load_from_df(df_ratings[['userID', 'placeID', 'rating']], reader)

# Dividir datos para entrenamiento y prueba
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Configurar algoritmo KNN con MSD y los mejores parámetros encontrados
algo = KNNBasic(k=20, sim_options={'name': 'msd', 'user_based': False})

# Entrenar el modelo
algo.fit(trainset)

# Función para recomendar lugares para un userID dado
def recommend_places(userID, algo, df_places, top_n=10):
    # Lista para almacenar recomendaciones
    recommendations = []
    
    # Predecir calificaciones para todos los lugares que el usuario no ha calificado
    place_ids = df_places['placeID'].unique()
    for place_id in place_ids:
        if not df_ratings[(df_ratings['userID'] == userID) & (df_ratings['placeID'] == place_id)].empty:
            continue  # Saltar lugares que el usuario ya ha calificado
        
        # Predecir la calificación para este lugar
        prediction = algo.predict(userID, place_id)
        recommendations.append((place_id, prediction.est))
    
    # Ordenar recomendaciones por calificación estimada
    recommendations.sort(key=lambda x: x[1], reverse=True)
    
    # Obtener las mejores recomendaciones
    top_recommendations = recommendations[:top_n]
    
    # Preparar detalles de las mejores recomendaciones
    top_recommendations_details = []
    for place_id, rating in top_recommendations:
        place_info = df_places[df_places['placeID'] == place_id].iloc[0]
        top_recommendations_details.append({
            'Nombre': place_info['name'],
            'Dirección': place_info['address'],
            'Calificación Estimada': rating
        })
    
    return top_recommendations_details

# Interfaz de usuario de Streamlit
st.title('Sistema de Recomendación de Restaurantes')

# Campo de entrada para userID
userID = st.text_input('Ingrese su userID (por ejemplo, U1077):')

# Botón de recomendación
if st.button('Obtener Recomendaciones'):
    if userID:
        # Mostrar recomendaciones
        recommendations = recommend_places(userID, algo, df_places)
        st.write(pd.DataFrame(recommendations))
    else:
        st.warning('Por favor ingrese un userID.')

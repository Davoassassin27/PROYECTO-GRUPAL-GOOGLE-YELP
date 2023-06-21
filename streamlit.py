import streamlit as st
from joblib import load
import pandas as pd

model_forest=load('modelo_RF3.joblib')
index_model=pd.read_csv('dataset/index_model3.csv')


st.title("Modelo de predicción de Rating ")
st.sidebar.header("Valores de entrada")

caracteristicas = ["Restaurant", "Food", "Bar"]
categoria = st.sidebar.selectbox("Selecciona una categoría:", caracteristicas)

ciudades = ["Albany", "Allen", "Waco"]
ciudad = st.sidebar.selectbox("Selecciona una ciudad:", ciudades)

st.write("Ciudad :", ciudad)
st.write("Categoría :", categoria)

def prediccion(categoria,ciudad):
    index_model=pd.read_csv('dataset/index_model3.csv')
    if ciudad in index_model.columns.tolist(): 
        if categoria in index_model.columns.tolist():
            for i in index_model:
                if i==categoria:
                    index_model[i]=1
            for i in index_model:
                if i==ciudad:
                    index_model[i]=1
            e=model_forest.predict(index_model)
            return round(((e[0]-0)*(5-1)/(1-0)+1),1)
        else:
            return 'Esa Categoria no existe'
    else:
        return 'Esa ciudad no existe'

if st.sidebar.button("Realizar predicción"):
        # Lógica para realizar la predicción o ejecutar una acción al presionar el botón
    resultado = prediccion(categoria,ciudad)
        #st.markdown("<p style='font-size:48px'>El Rating estimado es... {} Estrellas</p>".format(resultado), unsafe_allow_html=True)
    stars = '⭐️' * int(resultado)  # Genera un string con el número de estrellas correspondiente a 'resultado'

    if resultado > 3:
        st.markdown(f'<p style="font-size: 24px; color: green;">El Rating estimado es {resultado} {stars}</p>', unsafe_allow_html=True)
    else:
        st.markdown(f'<p style="font-size: 24px; color: red;">El Rating estimado es {resultado} {stars}</p>', unsafe_allow_html=True)




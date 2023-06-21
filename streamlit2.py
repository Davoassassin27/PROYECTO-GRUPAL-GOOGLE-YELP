import streamlit as st
from joblib import load
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import re
import mtranslate as mt
from tensorflow import keras

################################### Modelo Prediccion ########################################################

model_forest=load('modelo_RF3.joblib')
index_model=pd.read_csv('index_model3.csv')
df = pd.read_csv("ciudades_categorias.csv")

def prediccion(categoria, ciudad, categoria1, categoria2):
    
    
    if ciudad in index_model.columns.tolist(): 
        if categoria in index_model.columns.tolist():
            for i in index_model:
                if i == categoria:
                    index_model[i] = 1
                    
            for i in index_model:
                if i == ciudad:
                    index_model[i] = 1
                    
            if categoria1 is not None:
                if categoria1 in index_model.columns.tolist():
                    index_model[categoria1] = 1
                #else:
                 #   return f'Esa categoría "{categoria1}" no existe'
                    
            if categoria2 is not None:
                if categoria2 in index_model.columns.tolist():
                    index_model[categoria2] = 1
               # else:
                 #   return f'Esa categoría "{categoria2}" no existe'
                    
            e = model_forest.predict(index_model)
            return round(((e[0] - 0) * (5 - 1) / (1 - 0) + 1), 1)
        else:
            return 'Esa categoría no existe'
    else:
        return 'Esa ciudad no existe'

################################### Modelo Recomendacion ########################################################

dfrec = pd.read_csv("ML/dataset2.csv", low_memory=False)

label_encoder = LabelEncoder()
label_encoder.fit(dfrec['atributos'])

def hacer_predicciones4(modelo, datos_nuevos, top_n=10):
    datos_codificados = pd.get_dummies(datos_nuevos[['stars', 'categoria', 'state']])
    predicciones = modelo.predict(datos_codificados)
    top_clases = np.argsort(-predicciones)[:, :top_n]
    etiquetas_predichas = []
    
    for ejemplo in top_clases:
        atributos = label_encoder.inverse_transform(ejemplo)
        atributos_divididos = [re.sub(r'(?<!^)(?=[A-Z])', ' ', attr.split(': ')[0]) + ': ' + attr.split(': ')[1] for attr in atributos]
        atributos_traducidos = [mt.translate(attr, "es", "en") for attr in atributos_divididos]
        etiquetas_predichas.append(atributos_traducidos)
    
    return etiquetas_predichas

modelo_cargado = keras.models.load_model('ML/modelo_at_1')

################################### Main ########################################################

def main():
    st.title("Modelo de predicción de Rating ")
    st.sidebar.header("Valores de entrada")

    
    caracteristicas = ["None"] + df["NDescripcion"].unique().tolist()
   
    categoria = st.sidebar.selectbox("Selecciona una categoría:", caracteristicas,index=0, key="categoria1")
    categoria1 = st.sidebar.selectbox("Selecciona una categoría:", caracteristicas,index=0, key="categoria2")
    categoria2 = st.sidebar.selectbox("Selecciona una categoría:", caracteristicas,index=0, key="categoria3")

          
    ciudades = ["None"] + df["city"].unique().tolist()
    ciudad = st.sidebar.selectbox("Selecciona una ciudad:", ciudades, index=0, key="ciudad")

    st.write("Ciudad :", ciudad)
    
    categorias_seleccionadas = [categoria, categoria1, categoria2]
    categorias_seleccionadas = [cat for cat in categorias_seleccionadas if cat != "None"]
    
    

    if len(categorias_seleccionadas) >= 0:
      st.write("Categoría:", ", ".join(categorias_seleccionadas))
    else:
        st.write("Categoría :", ciudad)


    resultado = 0
    stars = '⭐️' * int(0)

    if st.sidebar.button("Realizar predicción"):
        # Lógica para realizar la predicción o ejecutar una acción al presionar el botón
        resultado = prediccion(categoria,ciudad,categoria1,categoria2)
        stars = '⭐️' * int(resultado)  # Genera un string con el número de estrellas correspondiente a 'resultado'


        if resultado > 3:
            st.write(f'<p style="font-size: 24px; color: green;">El Rating estimado es {resultado} {stars}</p>', unsafe_allow_html=True)
        else:
            st.write(f'<p style="font-size: 24px; color: red;">El Rating estimado es {resultado} {stars}</p>', unsafe_allow_html=True)



    if st.sidebar.button("Realizar Recomendacion"):
        
        resultado = prediccion(categoria,ciudad,categoria1,categoria2)
        stars = '⭐️' * int(resultado)  # Genera un string con el número de estrellas correspondiente a 'resultado'


        if resultado > 3:
            st.write(f'<p style="font-size: 24px; color: green;">El Rating estimado es {resultado} {stars}</p>', unsafe_allow_html=True)
        else:
            st.write(f'<p style="font-size: 24px; color: red;">El Rating estimado es {resultado} {stars}</p>', unsafe_allow_html=True)

        
        nuevos_datos = pd.DataFrame({'stars': resultado, 'categoria': categoria, 'state': ['California']})
                      
        
        etiquetas_predichas = hacer_predicciones4(modelo_cargado, nuevos_datos, top_n=10)
        etiquetas_predichas = pd.DataFrame(etiquetas_predichas)
     
        st.markdown("**Los Atributos Recomendados son:**")

        for columna, etiquetas in etiquetas_predichas.iteritems():
            for etiqueta in etiquetas:
                st.markdown(f"- {etiqueta}")


if __name__ == "__main__":
    main()
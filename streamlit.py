
import streamlit as st
from joblib import load
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import re
import mtranslate as mt
from tensorflow import keras

################################### Modelo Prediccion ########################################################

model_forest=load('ML/modelo_RF3.joblib')
index_model=pd.read_csv('dataset/index_model3.csv')
df = pd.read_csv("dataset/ciudades_categorias.csv")

@st.cache_data
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

dfrec = pd.read_csv("dataset/dataset2.csv", low_memory=False)

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
   
    caracteristicas = sorted(caracteristicas, reverse=False)
    caracteristicas.insert(0, "None")
    categoria = st.sidebar.selectbox("Selecciona una categoría:", caracteristicas,index=0, key="categoria1")
    categoria1 = st.sidebar.selectbox("Selecciona una categoría:", caracteristicas,index=0, key="categoria2")
    categoria2 = st.sidebar.selectbox("Selecciona una categoría:", caracteristicas,index=0, key="categoria3")

          
    ciudades = ["None"] + df["city"].unique().tolist()
    ciudades = sorted(ciudades, reverse=False)
    ciudades.insert(0, "None")
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
       
        try:
            
                          
            resultado = prediccion(categoria,ciudad,categoria1,categoria2)
            stars = '⭐️' * int(resultado)  


            if resultado > 3:
                st.write(f'<p style="font-size: 24px; color: green;">El Rating estimado es {resultado} {stars}</p>', unsafe_allow_html=True)
            else:
                st.write(f'<p style="font-size: 24px; color: red;">El Rating estimado es {resultado} {stars}</p>', unsafe_allow_html=True)
        except ValueError as e:
                
                st.write("Error Asegurese de cargar una categoria y una ciudad")

    rating = st.sidebar.slider("Selecciona un Rating:", 1.0, 5.0, step=0.1, value=1.0)

    if st.sidebar.button("Realizar Recomendacion"):
        try:
        
            resultado = prediccion(categoria,ciudad,categoria1,categoria2)
            stars = '⭐️' * int(resultado)  
            if resultado > 3:
                st.write(f'<p style="font-size: 24px; color: green;">El Rating estimado es {resultado} {stars}</p>', unsafe_allow_html=True)
            else:
                st.write(f'<p style="font-size: 24px; color: red;">El Rating estimado es {resultado} {stars}</p>', unsafe_allow_html=True)


            estado_ciudad = df.loc[df['city'] == ciudad, 'state']
            estado = estado_ciudad.iloc[0]
            #estado
            
            #st.write("el estado elegido es: ",estado)
            
            
            nuevos_datos = pd.DataFrame({'stars': rating, 'categoria': categoria, 'state': [estado]})
                      
        
            etiquetas_predichas = hacer_predicciones4(modelo_cargado, nuevos_datos, top_n=10)
            etiquetas_predichas = [[etiqueta.replace(': verdadero', '').replace(': Verdadero', '') for etiqueta in sublista] for sublista in etiquetas_predichas]
            etiquetas_predichas = [[etiqueta.replace(': cierto', '').replace(': Cierto', '') for etiqueta in sublista] for sublista in etiquetas_predichas]
            etiquetas_predichas = pd.DataFrame(etiquetas_predichas)
     
            st.markdown("**Los Atributos Recomendados son:**")

            for columna, etiquetas in etiquetas_predichas.iteritems():
                for etiqueta in etiquetas:
                    st.markdown(f"- {etiqueta}")
        except ValueError as e:
                
                st.write("Error Asegurese de cargar una categoria y una ciudad")


if __name__ == "__main__":
    main()
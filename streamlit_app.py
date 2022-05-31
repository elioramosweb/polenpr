import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image,ImageDraw,ImageOps
from tqdm import tqdm
import streamlit as st
from stqdm import stqdm
import pandas as pd
import plotly.express as px
import glob
import os
import pickle
import requests
from io import BytesIO



#st.set_page_config(page_title="polen",layout="wide")

######################################
## suprimir los warnings de tensorflow
######################################

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


## función para encontrar la diferencia entre dos listas

def diff(li1, li2):
    return list(set(li1) - set(li2)) + list(set(li2) - set(li1))

## función para ajustar las dimensiones para que sean compatibles
## con el modelo de CNN

def reshapeImage(img):
    img = np.array(img).astype('float32')/255
    img = img.reshape((1,50,50,1))
    return img


## función para hacer la clasificación de una imagen

def predictImage(img):
    predictions = model.predict(img)
    score = tf.nn.softmax(predictions[0])
    return predictions,class_names[np.argmax(score)]

## función para leer imagen en PIL 

def load_image(image_file):
	img = Image.open(image_file)
	return img


## funcion para hacer grafico de barras

def prob_barplot(df):
    
    fig = px.bar(df, x='especie', y='probabilidad',
                 color="probabilidad",
                 color_continuous_scale=px.colors.sequential.Reds,
                 orientation="v")

    fig.update_layout(margin={"r": 5, "t": 50, "l": 1, "b": 1},
                      title_text='Clasificación para las ' + str(df.shape[0])  
                      + ' especies con mayor probabilidad',
                      title_x=0.5)

    fig.update_xaxes(showgrid=True, 
                     showline=True,
                     linecolor='black',
                     gridwidth=0.5, 
                     gridcolor='gray',
                     mirror=True)
    
    fig.update_yaxes(showgrid=True, 
                     showline=True,
                     linecolor='black',
                     gridwidth=0.5, 
                     gridcolor='gray',
                     mirror=True)

    fig.update_layout({'paper_bgcolor': 'rgba(255, 255, 255, 255)'})

    fig.update_layout(xaxis_range=[0,10],
                      yaxis_range=[0,100],
                      font_color="black")    
    
    return fig 


## hacer crop para (en caso de que sea necesario) tener dimensiones cuadradas 

def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))

def crop_max_square(pil_img):
    return crop_center(pil_img, min(pil_img.size), min(pil_img.size))


#########
## titulo
######### 

st.title("Clasificador de Polen utilizando un Modelo de Redes Neuronales e Inteligencia Artificial")


## cargar modelo de clasificación 

model = keras.models.load_model('modelo51000.h5')



st.sidebar.image("polen_bonito.png")


st.sidebar.markdown("<hr>",unsafe_allow_html=True)

col1, col2 = st.columns(2)

###############
## SUBIR IMAGEN
############### 


st.sidebar.markdown("<b><font color='#f63366'>Subir imagen que se quiere clasificar...</font></b>",unsafe_allow_html=True)
image = st.sidebar.file_uploader("", type=["png", "jpg", "jpeg"])


st.sidebar.markdown("<hr>",unsafe_allow_html=True)


##############################################################
## SLIDER PARA FIJAR CANTIDAD DE ESPECIES EN GRAFICO DE BARRAS
############################################################## 

st.sidebar.markdown("<b><font color='#f63366'>Cantidad de especies para el gráfico de barras</font></b>",unsafe_allow_html=True)
cantEspecies = st.sidebar.slider('',
                         min_value=1,max_value=10,value=10,step=1)


####################################
## leemos diccionario con las clases
####################################

file = open("idConverter.pkl","rb")

class_temp = list(pickle.load(file).keys())

class_names = [val.replace("_","") for val in class_temp]

if image is not None:
    
    st.markdown("<hr>",unsafe_allow_html=True)

    imageOriginal = load_image(image)

    imageCrop = crop_max_square(imageOriginal)

    imageResize = ImageOps.grayscale(imageCrop).resize((50,50))

    col1.image(imageOriginal,width=250,caption='Imagen Original')
    col2.image(imageResize,width=250,caption='Imagen Gris')

    probs,classPredict = predictImage(reshapeImage(imageResize))


    ## DATAFRAME ##

    df = pd.DataFrame(list(zip(class_names, probs[0])),
               columns =['especie', 'probabilidad'])

    df = df.sort_values(by='probabilidad', ascending=False).head(cantEspecies)

    df["probabilidad"] *= 100

    ## GRAFICO DE BARRAS 

    fig = prob_barplot(df)
    
    st.plotly_chart(fig,width=200)

    st.markdown("<b>Clasificación por especie</b>",unsafe_allow_html=True)
    
    st.markdown("<p>Se muestran las <b>cinco posibles especies </b> de un total de <b>85</b> basadas en\
                la probabilidad que produce el modelo. Si la probabilidad mayor\
                es cercana a 100 las restantes imágenes podrían ser clasificaciones dudosas.</p>",
                unsafe_allow_html=True) 
                

    col = {}

    col[0],col[1],col[2],col[3],col[4] = st.columns(5)
    
    ##gitPath = "https://github.com/elioramosweb/plantasOne/blob/main/"

    ##gitPath = "https://github.com/elioramosweb/plantasOne/blob/main/a14_1.jpg?raw=true"

    gitPath= "https://raw.githubusercontent.com/elioramosweb/plantasOne/main/"
  

    # IMAGENES EN COLORES 


    for i in range(5):

        temp = str(list(df["especie"])[i])

        url = gitPath + temp + "_1.jpg"

        page = requests.get(url)

        fig2 = Image.open(BytesIO(page.content))

        #fig2 = load_image(glob.glob("./Plantas/" + temp + "/*.jpg")[0])

        col[i].image(fig2,width=100,caption=temp)
   
    # IMAGENES EN ESCALA DE GRIS 

    for i in range(5):

        temp = str(list(df["especie"])[i])

        url = gitPath + temp + "_1.jpg"

        page = requests.get(url)

        fig2 = Image.open(BytesIO(page.content))

        col[i].image(ImageOps.grayscale(fig2),width=100,caption=temp + "(Gris)")

else:
    
    st.markdown("<h5>Una vez suba la imagen los resultados van a aparecer aquí....</h5>",unsafe_allow_html=True)
    

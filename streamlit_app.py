import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image,ImageDraw,ImageOps,ImageEnhance
import streamlit as st
import pandas as pd
import plotly.express as px
import glob
import os
import pickle
import requests
from io import BytesIO



st.set_page_config(page_title="polen",layout="wide")

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
                 color_continuous_scale=px.colors.sequential.Redor,
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

#st.title("<font color='#f63366'>Clasificador de Polen</font>")

st.markdown("<h1><font color='#f63366'>Clasificador de Polen </font></h1>",unsafe_allow_html=True)


st.subheader("Utilizando un Modelo de Redes Neuronales Convolucionales")

## cargar modelo de clasificación 

model = keras.models.load_model('modelo51000.h5')


st.sidebar.image("polen_bonito.png")


st.sidebar.markdown("<hr>",unsafe_allow_html=True)

col1, col2 = st.columns(2)

############################
## checkbox de mostrar texto 
############################ 

mostrarTexto = st.sidebar.checkbox('Mostrar texto informátivo',value=True)


########################################
## checkbox de mostrar titulo de sección
########################################

mostrarSeccion= st.sidebar.checkbox('Mostrar texto de sección por especie',value=True)

################################
## checkbox de mostrar imágenes
################################ 

mostrarImagen = st.sidebar.checkbox('Mostrar imágenes',value=True)

#############################
## checkbox de mostrar barras
#############################

mostrarBarra = st.sidebar.checkbox('Mostrar Gráfico Barras',value=True)

############################
## checkbox de mostrar tabla
############################ 

mostrarTabla = st.sidebar.checkbox('Mostrar Tabla con clasificación',value=True)


########################################
## checkbox para utilizar image enhancer
########################################

mostrarEnhancer = st.sidebar.checkbox('Mejorar imagen',value=False)


st.sidebar.markdown("<hr>",unsafe_allow_html=True)

enhanceFactor = 1.0

if mostrarEnhancer: 
    st.sidebar.markdown("<b><font color='#f63366'>Factor para mejorar imagen. Oprima Enter luego de entrar valor número.</font></b>",unsafe_allow_html=True)
    st.sidebar.markdown("<p align='justify'> Si el factor es 1.0 <u>la imagen se queda igual</u>. Valores mayores de 1.0 implica mayor contraste, mayor brillo, y más colores.\
        Valores menores de 1.0 implica menor constraste, menor brillo, etc.</p>",unsafe_allow_html=True)
    enhanceFactor = st.sidebar.text_input('Factor para mejorar imagen', value='1.0')
    st.sidebar.markdown("<hr>",unsafe_allow_html=True)

###############
## SUBIR IMAGEN
############### 


st.sidebar.markdown("<b><font color='#f63366'>Subir imagen o imágenes que se quiere clasificar...</font></b>",unsafe_allow_html=True)

with st.form("my-form", clear_on_submit=True):
    images = st.sidebar.file_uploader("", type=["png", "jpg", "jpeg"],accept_multiple_files=True)

st.sidebar.markdown("<hr>",unsafe_allow_html=True)



####################################
## leemos diccionario con las clases
####################################

file = open("idConverter.pkl","rb")

class_temp = list(pickle.load(file).keys())

class_names = [val.replace("_","") for val in class_temp]


if mostrarTexto:

    st.markdown("<b><font color='#f63366'>Clasificación por especie</font></b>",unsafe_allow_html=True)
    st.markdown("<p>Se muestran las <b>cinco posibles especies </b> de un total de <b>85</b> basadas en\
                la probabilidad que produce el modelo. Si la probabilidad mayor\
                es cercana a 100 las restantes imágenes podrían ser clasificaciones dudosas.</p>",
                unsafe_allow_html=True) 


if mostrarBarra:
    st.sidebar.markdown("<b><font color='#f63366'>Cantidad de especies para el gráfico de barras</font></b>",unsafe_allow_html=True)
    cantEspecies = st.sidebar.slider('',min_value=1,max_value=10,value=10,step=1)

listaDF = []

if images is not None:

    my_bar = st.progress(0)

    largo = len(images)

    inx = 1

    for image in images:

        ## ajustar dimensiones de imagen de entrada 

        imageOriginal = load_image(image)

        tt = ImageEnhance.Contrast(imageOriginal).enhance(float(enhanceFactor))

        imageCrop = crop_max_square(tt)

        imageResize = ImageOps.grayscale(imageCrop).resize((50,50))
 
        ## hacer predicción con el modelo 

        probs,classPredict = predictImage(reshapeImage(imageResize))
    
        ## DATAFRAME con resultados ##
    
        df = pd.DataFrame(list(zip(class_names, probs[0])),
                   columns =['especie', 'probabilidad'])
    
        ## slider para cantidad de barras 

        cantEspecies = 10

        df = df.sort_values(by='probabilidad', ascending=False).head(cantEspecies)
    
        df["probabilidad"] *= 100
        
        ## MOSTRAR TITULO DE SECCION 

        if mostrarSeccion:
            st.markdown("<hr>",unsafe_allow_html=True)
            st.markdown("<b><font color='#f63366'>"  + image.name + "</font></b>",unsafe_allow_html=True)        

        ## GRAFICO DE BARRAS 
    
        if mostrarBarra:
            fig = prob_barplot(df)
            st.plotly_chart(fig,width=200)
    

        lista1 = list(df["especie"])
        lista2 = list(df["probabilidad"])
                      
        temp = {lista1[i]:[lista2[i]] for i in range(5)}

        df2 = pd.DataFrame(temp).rename({0:'probabilidad'}).style.format("{:.2f}")

        col = {}
    
        col[0],col[1],col[2],col[3],col[4],col[5],col[6] = st.columns(7)
        
        gitPath= "https://raw.githubusercontent.com/elioramosweb/plantasOne/main/"
      

        # IMAGENES EN COLORES 
    
        if mostrarImagen:

            col[0].image(imageOriginal,width=100,caption=image.name)
            col[1].image("flecha.png",width=100)

        listaImagenes = []

        if enhanceFactor == 1.0:
            listaImagenes.append(image.name)
        else:
            listaImagenes.append(image.name + " enhance(" + str(enhanceFactor) + ")")
    
        for i in range(5):
    
            tempEspecie = str(list(df["especie"])[i])
           
            tempProb = str(round(list(df["probabilidad"])[i],2))
    
            url = gitPath + tempEspecie + "_1.jpg"
    
            page = requests.get(url)
    
            fig2 = Image.open(BytesIO(page.content))

            imageCaption = tempEspecie + " - " + tempProb + "%"

            listaImagenes.append(imageCaption)

            if mostrarImagen:
                col[i+2].image(fig2,width=100,caption=imageCaption)
       
        ##st.markdown("<hr>",unsafe_allow_html=True)

        my_bar.progress(round(100*(inx/largo)))

        inx += 1

        listaDF.append(listaImagenes)

    ## creacion de archivo de salida 
    
    #print(listDF)
    
    if len(listaDF) > 0:

        dfOutput = pd.DataFrame(listaDF)

        dfOutput.columns = ['imagen','p1','p2','p3','p4','p5']
        
        dfOutput = dfOutput.sort_values(by='imagen')

        dfOutput = dfOutput.drop_duplicates( keep='last')

        if mostrarTabla:
            st.markdown("<hr>",unsafe_allow_html=True)
            st.markdown("<b><font color='#f63366'>Tabla de resultados</font></b>",unsafe_allow_html=True)
            st.dataframe(dfOutput.sort_index())

        @st.cache
        def convert_df(df):
            return df.to_csv().encode('utf-8')

        if mostrarTabla:

            csv = convert_df(dfOutput)
            st.download_button(
            "Presione para bajar tabla de resultados en CSV",
            csv,
            "resultados.csv",
            "text/csv",
            key='download-csv'
            ) 

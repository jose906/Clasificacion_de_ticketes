# Proyecto2
# NLP - Clasificacion Automatica de Tickets

![image](https://user-images.githubusercontent.com/15108160/221388554-527a635e-a4d4-4d3a-a821-520b6dbaf149.png)
 Clasificacion Automatica de Tickets con NLP
Integrantes
José Estensoro (josee906@gmail.com)
Roger Patón (oviroger@gmail.com)
Descripcion del Problema
Debe crear un modelo que pueda clasificar las quejas (complaints) de los clientes en función de los productos/servicios. Al hacerlo, puede segregar estos tickets en sus categorías relevantes y, por lo tanto, ayudar en la resolución rápida del problema.

Realizará el modelado de temas en los datos .json proporcionados por la empresa. Dado que estos datos no están etiquetados, debe aplicar NMF para analizar patrones y clasificar los tickets en los siguientes cinco grupos según sus productos/servicios:

Tarjetas de Credito / Tarjetas Prepagadas (Credit card / Prepaid Card)

Servicios de Cuentas de Banco (Bank account services)

Reportes de Robos (Theft/Dispute reporting)

Prestamos Hipotecarios y Otros Prestamos (Mortgages/loans)

Otros

Con la ayuda del modelado de temas, podrá asignar cada ticket a su respectivo departamento/categoría. Luego puede usar estos datos para entrenar cualquier modelo supervisado, como regresión logística, árbol de decisión o bosque aleatorio. Usando este modelo entrenado, puede clasificar cualquier nuevo ticket de soporte de quejas de clientes en su departamento correspondiente.


# Librerias 
import json
import numpy as np
import pandas as pd
import re, string

# Import NLTK libraries
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Import Spacy libraries
import spacy
import en_core_web_sm
nlp = en_core_web_sm.load()

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

from plotly.offline import plot
import plotly.graph_objects as go
import plotly.express as px

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from pprint import pprint

# Suppressing Warnings
import warnings
warnings.filterwarnings('ignore')

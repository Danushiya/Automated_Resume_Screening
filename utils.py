import altair as alt
# import datetime
import joblib
import nltk
import numpy as np
import pandas as pd
import re
import streamlit as st 
import time
import pdfkit
import os
import PyPDF2  # Required for reading PDF files
import docx  # Required for reading DOCX files
import pandas as pd
import io
import PyPDF2
import pytesseract
from PIL import Image
import docx2txt
import tempfile
import subprocess
import textract
import matplotlib.pyplot as plt
from gensim.corpora import Dictionary
from gensim.models import KeyedVectors, TfidfModel
from gensim.similarities import SoftCosineSimilarity, SparseTermSimilarityMatrix
from gensim.similarities.annoy import AnnoyIndexer
from io import BytesIO
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from pandas.api.types import is_categorical_dtype, is_numeric_dtype
from PIL import Image
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()


def read_resumes(file):
    resumes = []

    if file.type == 'application/pdf':
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
        for page_num in range(len(pdf_reader.pages)):
            text = pdf_reader.pages[page_num].extract_text()
            resumes.append(text)

    elif file.type in ['image/png', 'image/jpeg', 'image/jpg']:
        image = Image.open(io.BytesIO(file.read()))
        text = image_to_text(image)
        resumes.append(text)

    elif file.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        text = docx2txt.process(io.BytesIO(file.read()))
        resumes.append(text)

    elif file.type == 'text/plain':
        text = file.read().decode('utf-16', 'ignore')
        resumes.append(text)

    elif file.type == 'application/msword':
        # If textract cannot handle DOC files, consider alternatives like antiword:
        # text = textract.process(file.read())
            # Assuming antiword is installed
        process = subprocess.Popen(['antiword', '-'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        output, _ = process.communicate(file.read())
        text = output.decode('utf-8')
        resumes.append(text)


    elif file.type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
        excel_df = pd.read_excel(io.BytesIO(file.read()))
        for column in excel_df.columns:
            resumes.extend(excel_df[column].astype(str))

    return pd.DataFrame({'Resume': resumes})

def image_to_text(image):
    # Convert image to text using OCR (Optical Character Recognition)
    text = pytesseract.image_to_string(image)
    return text

def clickRank():
    st.session_state.processRank = True

def convertDfToXlsx(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine = 'xlsxwriter')
    df.to_excel(writer, index = False, sheet_name = 'Sheet1')
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    format1 = workbook.add_format({'num_format': '0.00'}) 
    worksheet.set_column('A:A', None, format1)  
    writer.close()
    processedData = output.getvalue()
    return processedData

def filterDataframeClf(df: pd.DataFrame) -> pd.DataFrame:
    modify = st.toggle("Add filters", key = 'filter-clf-1')
    if not modify:
        return df
    df = df.copy()
    modificationContainer = st.container()
    with modificationContainer:
        toFilterColumns = st.multiselect("Filter table on", df.columns, key = 'filter-clf-2')
        for column in toFilterColumns:
            left, right = st.columns((1, 20))
            left.write("↳")
            widgetKey = f'filter-clf-{toFilterColumns.index(column)}-{column}'
            if is_categorical_dtype(df[column]):
                userCatInput = right.multiselect(
                    f'Values for {column}',
                    df[column].unique(),
                    default = list(df[column].unique()),
                    key = widgetKey 
                )
                df = df[df[column].isin(userCatInput)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                userNumInput = right.slider(
                    f'Values for {column}',
                    min_value = _min,
                    max_value = _max,
                    value = (_min, _max),
                    step = step,
                    key = widgetKey 
                )
                df = df[df[column].between(*userNumInput)]
            else:
                userTextInput = right.text_input(
                    f'Substring or regex in {column}',
                    key = widgetKey 
                )
                if userTextInput:
                    userTextInput = userTextInput.lower()
                    df = df[df[column].astype(str).str.lower().str.contains(userTextInput)]
    return df

def filterDataframeRnk(df: pd.DataFrame) -> pd.DataFrame:
    modify = st.toggle("Add filters", key = 'filter-rnk-1')
    if not modify:
        return df
    df = df.copy()
    modificationContainer = st.container()
    with modificationContainer:
        toFilterColumns = st.multiselect("Filter table on", df.columns, key = 'filter-rnk-2')
        for column in toFilterColumns:
            left, right = st.columns((1, 20))
            left.write("↳")
            widgetKey = f'filter-rnk-{toFilterColumns.index(column)}-{column}'
            if is_categorical_dtype(df[column]):
                userCatInput = right.multiselect(
                    f'Values for {column}',
                    df[column].unique(),
                    default = list(df[column].unique()),
                    key = widgetKey
                )
                df = df[df[column].isin(userCatInput)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                userNumInput = right.slider(
                    f'Values for {column}',
                    min_value = _min,
                    max_value = _max,
                    value = (_min, _max),
                    step = step,
                    key = widgetKey
                )
                df = df[df[column].between(*userNumInput)]
            else:
                userTextInput = right.text_input(
                    f'Substring or regex in {column}',
                    key = widgetKey
                )
                if userTextInput:
                    userTextInput = userTextInput.lower()
                    df = df[df[column].astype(str).str.lower().str.contains(userTextInput)]
    return df

def getWordnetPos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    
def performLemmatization(text):
    text = re.sub('http\S+\s*', ' ', text)
    text = re.sub('RT|cc', ' ', text)
    text = re.sub('#\S+', '', text)
    text = re.sub('@\S+', '  ', text)
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', text)
    text = re.sub(r'[^\x00-\x7f]',r' ', text)
    text = re.sub('\s+', ' ', text)
    words = word_tokenize(text)
    words = [
        lemmatizer.lemmatize(word.lower(), pos = getWordnetPos(pos)) 
        for word, pos in pos_tag(words) if word.lower() not in stop_words
    ]
    return words

def performStemming(text):
    text = re.sub('http\S+\s*', ' ', text)
    text = re.sub('RT|cc', ' ', text)
    text = re.sub('#\S+', '', text)
    text = re.sub('@\S+', '  ', text)
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', text)
    text = re.sub(r'[^\x00-\x7f]',r' ', text)
    text = re.sub('\s+', ' ', text)
    words = word_tokenize(text)
    words = [stemmer.stem(word.lower()) for word in words if word.lower() not in stop_words]
    text = ' '.join(words)
    return text 

def loadModel():
    model_path = 'svm_model.joblib'
    model = joblib.load(model_path)
    return model

model = loadModel()

def rankResumes(text, df):
    progressBar = st.progress(0)
    progressBar.progress(0, text="Preprocessing data ...")
    startTime = time.time()
    jobDescriptionText = performLemmatization(text)
    df['cleanedResume'] = df['Resume'].apply(lambda x: performLemmatization(x))
    documents = [jobDescriptionText] + df['cleanedResume'].tolist()
    progressBar.progress(13, text="Creating a dictionary ...")
    dictionary = Dictionary(documents)
    progressBar.progress(25, text="Creating a TF-IDF model ...")
    tfidf = TfidfModel(dictionary=dictionary)
    progressBar.progress(38, text="Calculating TF-IDF vectors...")
    tfidf_vectors = tfidf[[dictionary.doc2bow(resume) for resume in df['cleanedResume']]]
    query_vector = tfidf[dictionary.doc2bow(jobDescriptionText)]
    
    # Load the SVM model
    svm_model = joblib.load('svm_model.joblib')

    progressBar.progress(50, text="Calculating similarity scores...")
    similarities = []
    for vector in tfidf_vectors:
        # Convert query_vector and vector to numpy arrays
        query_vector_array = np.array(query_vector)
        vector_array = np.array(vector)
        
        # Ensure both arrays have the same number of dimensions
        if query_vector_array.ndim != vector_array.ndim:
            vector_array = np.expand_dims(vector_array, axis=0)
        
        # Calculate cosine similarity
        similarity = cosine_similarity(query_vector_array, vector_array)
        similarities.append(similarity[0][0])

    progressBar.progress(75, text="Ranking resumes...")
    df['Similarity Score (-1 to 1)'] = similarities
    df['Rank'] = df['Similarity Score (-1 to 1)'].rank(ascending=False, method='dense').astype(int)
    df.sort_values(by='Rank', inplace=True)
    df.drop(columns=['cleanedResume'], inplace=True)
    
    endTime = time.time()
    elapsedSeconds = endTime - startTime
    hours, remainder = divmod(int(elapsedSeconds), 3600)
    minutes, _ = divmod(remainder, 60)
    secondsWithDecimals = '{:.2f}'.format(elapsedSeconds % 60)
    elapsedTimeStr = f'{hours} h : {minutes} m : {secondsWithDecimals} s'
    
    progressBar.progress(100, text=f'Ranking Complete!')
    time.sleep(1)
    progressBar.empty()
    
    st.info(f'Finished ranking {len(df)} resumes - {elapsedTimeStr}')
    return df 

def clickClassify():
    # Function to handle button click event
    st.session_state.processClf = True

def addZeroFeatures(matrix):
    maxFeatures = 18038
    numDocs, numTerms = matrix.shape
    missingFeatures = maxFeatures - numTerms
    if missingFeatures > 0:
        zeroFeatures = csr_matrix((numDocs, missingFeatures), dtype=np.float64)
        matrix = hstack([matrix, zeroFeatures])
    return matrix

def create_bar_chart(df):
    valueCounts = df['Industry Category'].value_counts().reset_index()
    valueCounts.columns = ['Industry Category', 'Count']
    newDataframe = pd.DataFrame(valueCounts)
    barChart = alt.Chart(newDataframe,
    ).mark_bar(
        color = '#56B6C2',
        size = 13 
    ).encode(
        x = alt.X('Count:Q', axis = alt.Axis(format = 'd'), title = 'Number of Resumes'),
        y = alt.Y('Industry Category:N', title = 'Category'),
        tooltip = ['Industry Category', 'Count']
    ).properties(
        title = 'Number of Resumes per Category',
    )
    return barChart

def loadTfidfVectorizer():
    tfidfVectorizerFileName = f'tfidf_vectorizer.joblib' 
    return joblib.load(tfidfVectorizerFileName)

def loadLabelEncoder():
    labelEncoderFileName = f'label_encoder.joblib'
    return joblib.load(labelEncoderFileName)

def cleanResume(resumeText):
    resumeText = re.sub('http\S+\s*', ' ', resumeText)  # remove URLs
    resumeText = re.sub('RT|cc', ' ', resumeText)  # remove RT and cc
    resumeText = re.sub('#\S+', '', resumeText)  # remove hashtags
    resumeText = re.sub('@\S+', '  ', resumeText)  # remove mentions
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)  # remove punctuations
    resumeText = re.sub(r'[^\x00-\x7f]',r' ', resumeText) 
    resumeText = re.sub('\s+', ' ', resumeText)  # remove extra whitespace

    words = resumeText.split()
    words = [word for word in words if word.lower() not in stop_words]
    words = [stemmer.stem(word.lower()) for word in words if word.lower() not in stop_words]
    resumeText = ' '.join(words)
    return resumeText

def classifyResumes(df):
    progressBar = st.progress(0)
    progressBar.progress(0, text="Preprocessing data ...")
    startTime = time.time()
    
    df['cleanedResume'] = df['Resume'].apply(lambda x: performStemming(x))
    resumeText = df['cleanedResume'].values
    
    progressBar.progress(20, text="Extracting features ...")
    vectorizer = loadTfidfVectorizer()
    wordFeatures = vectorizer.transform(resumeText)
    wordFeaturesWithZeros = addZeroFeatures(wordFeatures)
    
    # Load the SVM model using joblib
    svm_model = joblib.load('svm_model.joblib')
    
    progressBar.progress(60, text="Predicting categories ...")
    # Predict categories using the loaded SVM model
    predictedCategories = svm_model.predict(wordFeaturesWithZeros)
    
    progressBar.progress(80, text="Finishing touches ...")
    
    # Load the LabelEncoder
    le = loadLabelEncoder()
    

    # Assign a default label for unseen categories
    default_label = 'Unknown'
    predictedLabels = [default_label if label not in le.classes_ else label for label in predictedCategories]
    
    df['Industry Category'] = predictedLabels
    df['Industry Category'] = pd.Categorical(df['Industry Category'])
    df.drop(columns=['cleanedResume'], inplace=True)
    
    endTime = time.time()
    elapsedSeconds = endTime - startTime
    hours, remainder = divmod(int(elapsedSeconds), 3600)
    minutes, _ = divmod(remainder, 60)
    secondsWithDecimals = '{:.2f}'.format(elapsedSeconds % 60)
    elapsedTimeStr = f'{hours} h : {minutes} m : {secondsWithDecimals} s'
    
    progressBar.progress(100, text=f'Classification Complete!')
    time.sleep(1)
    progressBar.empty()
    st.info(f'Finished classifying {len(resumeText)} resumes - {elapsedTimeStr}')
    
    return df
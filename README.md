# AUTOMATED RESUME SCREENING

**Efficiently screen and categorize resumes using SVM-based Resume Screening Tool with Streamlit UI**

## Overview
The Automated Resume Screening Tool is a machine learning-based application designed to streamline the initial screening process for HR teams. By leveraging Support Vector Machine (SVM) algorithm and natural language processing techniques, this tool efficiently matches job descriptions with candidate resumes, categorizing them based on their suitability for specific job roles.

## Objectives
- **Developing an AI tool for Resume screening:** The primary objective of this project is to create an AI-powered tool that automates the resume screening process, reducing manual effort and time required by HR teams.

- **Matching job descriptions with candidate resumes:** The tool aims to analyze both job descriptions and candidate resumes to identify relevant skills, experiences, and qualifications, thereby ensuring a better match between candidates and job roles.

- **Streamlining initial screening:** By automating the screening process, the tool helps HR teams quickly filter through a large volume of resumes, allowing them to focus their time and attention on the most promising candidates.

## How it Works
**1. Model Training:**

- Open the provided Jupyter Notebook.
- Execute each cell to run the code for model training.
- View the results and outputs for insights into the trained model.
  
**2. Implementing the Model:**

- Open Visual Studio Code editor.
- Run the **main.py** file.
- Execute the command **streamlit run FILE PATH** in the Command Prompt to launch the Streamlit UI.
  
## Dependencies
  <div align="center">
  <img src="https://techstack-generator.vercel.app/python-icon.svg" alt="icon" width="50" height="50" />
</div>

- Python 3.7 or higher
- pandas
- mlxtend
- altair
- nltk
- numpy
- streamlit
- pdfkit
- PyPDF2
- docx
- pytesseract
- Pillow (Python Imaging Library)
- docx2txt
- textract
- matplotlib
- gensim
  
These dependencies can be installed using **pip install -r requirements.txt** as specified in the code comment.

## Dataset Description
**Source:**  [Hugging Face: Resume_classification_dataset](https://huggingface.co/datasets/Saba06huggingface/Resume_classification_dataset)

**Description:** The dataset contains resume details (professional experience, accomplishments, skills, and education) along with job role categories.

## Usage
1. **Model Training:**

- Open the provided Jupyter Notebook and execute each cell to train the model.
  
2. **Implementing the Model:**

- Run the **main.py** file in a Python environment.
- Access the Streamlit UI by executing **streamlit run FILE PATH** in the Command Prompt.

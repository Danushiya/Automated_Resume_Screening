import pandas as pd
import streamlit as st
from io import BytesIO
from utils import *

backgroundPattern = """
<style>
[data-testid="stAppViewContainer"] {
background-color: #FFFFFF;
opacity: 1;
background-image: radial-gradient(#D1D1D1 0.75px, #FFFFFF 0.75px);
background-size: 15px 15px;
}
</style>
"""
st.markdown(backgroundPattern, unsafe_allow_html=True)

st.write("""
# RESUME SCREENING TOOL
""")
st.caption("""
Using Support Vector Machine (SVM) algorithm
######
""")

tab1, tab2 = st.tabs(['Rank', 'Classify'])

with tab1:
    st.header('Rank Resumes')
    st.caption("Upload Job Description and Resumes")
    description_option = st.radio("Select how to provide the job description:", ("Paste or type the job description", "Upload a file"))

    if description_option == "Paste or type the job description":
        job_description_text = st.text_area("Paste or type the job description here:")
        uploaded_job_description = io.BytesIO(job_description_text.encode('utf-8'))  # Convert text to BytesIO object
    else:
        uploaded_job_description = st.file_uploader('Upload Job Description', type=[ 'txt'], key="upload-file-description")

    uploaded_resume_rank = st.file_uploader('Upload Resumes', type=['xlsx', 'pdf', 'docx', 'txt', 'png','jpeg','jpg'], key="upload-file-1", accept_multiple_files=True)

    if uploaded_job_description and uploaded_resume_rank:
        is_button_disabled_rank = False
    else:
        st.session_state.process_rank = False
        is_button_disabled_rank = True

    if 'process_rank' not in st.session_state:
        st.session_state.process_rank = False

    if st.button('Match Resumes', disabled=is_button_disabled_rank):
        st.session_state.process_rank = True

    if st.session_state.process_rank:
        st.divider()
        st.header('Ranking Output')
        job_description_rank = uploaded_job_description.read().decode('utf-8', 'ignore')
        ranked_resumes = pd.DataFrame()
        for resume_file in uploaded_resume_rank:
            resume_rank = read_resumes(resume_file)
            ranked_resumes = pd.concat([ranked_resumes, resume_rank], ignore_index=True)

        ranked_resumes = rankResumes(job_description_rank, ranked_resumes)
        with st.expander('View Job Description'):
            st.write(job_description_rank)
        current_rank = filterDataframeRnk(ranked_resumes)
        st.dataframe(current_rank, use_container_width=True, hide_index=True)
        xlsx_rank = convertDfToXlsx(current_rank)
        st.download_button(label='Save Ranked Output as XLSX', data=xlsx_rank, file_name='Resumes_ranked.xlsx')

with tab2:
    st.header('Classify Resumes')
    st.caption("Upload Resumes for Classification")

    uploaded_resume_clf = st.file_uploader('Upload Resumes', type=['xlsx', 'pdf', 'doc', 'docx', 'txt','png','jpeg','jpg'], key="upload-file-2", accept_multiple_files=True)

    if uploaded_resume_clf:
        is_button_disabled_clf = False
    else:
        st.session_state.process_clf = False
        is_button_disabled_clf = True

    if 'process_clf' not in st.session_state:
        st.session_state.process_clf = False

    if st.button('Classify Resumes', disabled=is_button_disabled_clf):
        st.session_state.process_clf = True

    if st.session_state.process_clf:
        st.divider()
        st.header('Classification Output')

        if uploaded_resume_clf:
            classified_resumes = pd.DataFrame()
            for resume_file in uploaded_resume_clf:
                resume_df = read_resumes(resume_file)
                if resume_df is not None:
                    classified_resumes = pd.concat([classified_resumes, resume_df], ignore_index=True)

            if len(classified_resumes) > 0:
                classified_resumes = classifyResumes(classified_resumes)
                with st.expander('View Classification Results'):
                    bar_chart = create_bar_chart(classified_resumes)
                    st.altair_chart(bar_chart, use_container_width=True)
                current_df_clf = filterDataframeClf(classified_resumes)
                st.dataframe(current_df_clf, use_container_width=True, hide_index=True)
                xlsx_clf = convertDfToXlsx(current_df_clf)
                st.download_button(label='Save Classified Output as XLSX', data=xlsx_clf, file_name='Resumes_classified.xlsx')
            else:
                st.error("Oops! Something went wrong while reading the resumes. Please make sure resumes are uploaded in the correct format.")
                
# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
import pandas as pd 
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)

OPENAI_API_KEY = st.secrets['OPENAI_API_KEY']


def create_agent():
    df = pd.read_csv('df_Last5Juz.csv')
    prefix = """
    You are an advanced agent designed to assist in Quran memorization by identifying mutashabihat (similar verses) with an emphasis on precision and relevance. You have been given a dataframe. This has for every verse in the Quran, the top 5 most similar verses and their similarity score.

    For all questions use the csv 'df_Last5Juz.csv'.

    You will also be asked to index and query the dataframe. For example: give me all the ayas that have inna muttaqeen. When presented with this type of question first (if needed) convert the text into arabic, then remove the diacritical marks and search in the column 'ayah_ar_no_tashkeel'. 

    Whenever presenting an aya, format it in the manner. 
    Ayah: 55:22
    Arabic: مِنْهُمَا يَخْرُجُ اللُّؤْلُؤُ وَالْمَرْجَانُ
    Translation: [translation]

    Limit your results to at most 10 examples, unless the user specifies otherwise. Order the results by relevance, highlighting the most significant similarities first even if they are asking for the whole surah. Prioritize the most similar scores. 
    """
    llm = ChatOpenAI(openai_api_key = OPENAI_API_KEY, model = 'gpt-4-turbo-preview', temperature=0)
    agent_executor = create_pandas_dataframe_agent(
        llm,
        df,
        prefix = prefix,
        agent_type="openai-tools",
    )
    return agent_executor

def retrieval_answer(query):
    agent_executor = create_agent()
    answer = agent_executor.invoke(query)
    return answer['output']


def main():
    st.title("Quran Memorization Helper")
    with st.expander("Learn More About the Project"):
        st.write("""
        ### Project Goal
        To assist in memorizing the Quran by finding and presenting verses similar to the one the user is currently memorizing. This can help in understanding the context and variations in Quranic themes and aid in memorization by association.

        ### Project Information
        This application is designed to support Quran memorization efforts by retrieving and displaying verses similar to the user's current focus. It utilizes a database of Quranic verses and advanced search techniques to find related ayahs (verses).

        #### Technologies Used
        - **LLM (Large Language Models)**: For processing queries and understanding the context of the verses.
        - **BERT Encoding and Cosine Similarity**: Determining which verses are similar
        - **Streamlit**: For creating the web interface of the application.

        #### How it Works
        - The user inputs a verse or keywords from a verse they are memorizing.
        - The app processes the query to understand the context and searches for similar verses.
        - The relevant verses are then displayed on the app interface, helping the user in memorization by providing related context and themes.

        Learning more by reading this article: https://medium.com/@gasperjw/hifdh-helper-an-assistant-to-quran-memorization-c25eddda49d0
        """)
    query = st.text_input("Enter the verse or keywords you are memorizing...") 
    ask_button = st.button("Find Similar Verses")
    
    if query and (ask_button or query != ""):
        st.info("Your Input: " + query)
        similar_verses = retrieval_answer(query)
        st.success(similar_verses)



if __name__ == "__main__":
    main()

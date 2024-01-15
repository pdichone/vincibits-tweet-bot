import os
from dotenv import find_dotenv, load_dotenv
import openai
from langchain_openai import (
    ChatOpenAI,
)  # This is the class that we will use to interact with OpenAI

from langchain.prompts import PromptTemplate
from langchain.chains import SequentialChain, LLMChain


import streamlit as st

load_dotenv(find_dotenv())

openai.api_key = os.getenv("OPENAI_API_KEY")

st.title("Tweet Generator")
st.header("Generate Your Next Tweet with Tweet Bot")

st.text("1. Describe Your Tweet (Or Copy and Paste and Existing One)")
description = st.text_area("", height=14)

st.text("2. Select Your Voice")
voice_options = st.multiselect(
    label="select", options=["Casual", "Professional", "Funny"]
)

llm = ChatOpenAI(temperature="0.9")

prompt = PromptTemplate(
    input_variables=["descripton", "option"],
    template="""Write me a very {option} tweet that is based on this description: {description}.
    Include two appropriate emojis and hashtags at the end of the tweet""",
)
prompt_two = PromptTemplate(
    input_variables=["descripton", "option"],
    template="""Write me another, different and very {option} tweet that is based on this description: {description}.
    Include two appropriate emojis and hashtags at the end of the tweet""",
)


tweet = prompt | llm
tweet_two = prompt_two | llm

if st.button("Generate Tweet") and description and voice_options:
    tweet = tweet.invoke({"description": description, "option": voice_options})
    tweet_two = tweet_two.invoke({"description": description, "option": voice_options})
    # print(tweet)
    st.info(tweet.content)
    st.divider()
    st.info(tweet_two.content)

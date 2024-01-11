import os
from dotenv import find_dotenv, load_dotenv
import openai
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import SequentialChain, LLMChain


import streamlit as st

load_dotenv(find_dotenv())

openai.api_key = os.getenv("OPENAI_API_KEY")

st.title("Tweet Generator")
st.header("Generate Your Next Tweet with Tweet Bot")

st.text("1. Describe Your Tweet (Or Copy and Paste and Existing One)")
prompt = st.text_area("", height=14)

st.text("2. Select Your Voice")
voice_options = st.multiselect(label="", options=["Casual", "Professional", "Funny"])

llm = ChatOpenAI(temperature="0.9")

tweet_one_template = PromptTemplate(
    input_variables=["description", "option"],
    template="""Write me a very {option} tweet that is based on this description: {description}. 
    Include two appropriate emojis and hashtags at the end of the tweet""",
)

tweet_two_template = PromptTemplate(
    input_variables=["description", "option"],
    template="""Write me a very {option} tweet that is based on {description}. 
    Include two appropriate emojis and hashtags at the end of the tweet""",
)

tweet_one_chain = LLMChain(
    llm=llm, prompt=tweet_one_template, verbose=True, output_key="tweet_one"
)
tweet_two_chain = LLMChain(
    llm=llm, prompt=tweet_two_template, verbose=True, output_key="tweet_two"
)

#  We combine all of these chains into a sequential chain
sequential_chain = SequentialChain(
    chains=[tweet_one_chain, tweet_two_chain],
    input_variables=["description", "option"],
    verbose=True,
)


if st.button("Generate Tweets") and prompt and voice_options:
    response = sequential_chain.invoke({"description": prompt, "option": voice_options})
    # response = sequential_chain({"description": prompt, "option": voice_options})
    # print(response)
    # print(f"Tweet 1: {response['tweet_one']} -- Tweet2:{response['tweet_two']}")

    st.divider()
    st.info(response["tweet_two"])

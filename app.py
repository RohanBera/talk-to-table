import streamlit as st
import pandas as pd
import os
from pandasai import SmartDataframe

# from pandasai.callbacks import BaseCallback
from pandasai.llm.local_llm import LocalLLM
from pandasai.llm import BambooLLM
from pandasai.responses.response_parser import ResponseParser

# for bamboo LLM
model = BambooLLM(api_key="")

# for OLLAMA models
# model = LocalLLM(api_base="http://localhost:11434/v1", model="llama3:8b")


class StreamlitResponse(ResponseParser):
    def __init__(self, context) -> None:
        super().__init__(context)

    def format_dataframe(self, result):
        st.dataframe(result["value"])
        return

    def format_plot(self, result):
        st.image(result["value"])
        return

    def format_other(self, result):
        st.write(result["value"])
        return


st.write("# Chat with Titanic Dataset 🦙")

df = pd.read_csv("titanic.csv")

query_engine = SmartDataframe(
    df,
    config={
        "llm": model,
        "response_parser": StreamlitResponse,
    },
)

st.write("🔎 Dataframe Preview")
st.write(df.tail(20))

query = st.text_area("🗣️ Chat with Dataframe")

if st.button("Generate"):
    if query:
        with st.spinner("Generating Response..."):

            if os.path.exists("exports/charts/temp_chart.png"):
                os.remove("exports/charts/temp_chart.png")

            st.write(query_engine.chat(query))

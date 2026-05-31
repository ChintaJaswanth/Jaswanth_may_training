pip install streamlit google-generativeai

import streamlit as st
import google.generativeai as genai
import os

# Configure Gemini API
# Check if running in Colab to use google.colab.userdata
# Otherwise, try st.secrets or st.text_input
API_KEY = ""
try:
    from google.colab import userdata
    API_KEY = userdata.get('GEMINI_API_KEY')
except ImportError:
    # Not in Colab, try Streamlit secrets or prompt user
    API_KEY = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY"))

if not API_KEY:
    st.warning("GEMINI_API_KEY not found. Please enter it below or set it in Colab secrets / Streamlit secrets.toml")
    API_KEY = st.text_input("Enter Gemini API Key", type="password", help="You can get your API key from https://makersuite.google.com/key")
    if not API_KEY:
        st.stop()

genai.configure(api_key=API_KEY)

model = genai.GenerativeModel("gemini-pro-latest") # Using gemini-pro-latest as identified earlier

st.set_page_config(page_title="AI Blog & Email Generator", layout="wide")

st.title("✍️ AI Blog & Email Generator")

content_type = st.radio(
    "Select Content Type",
    ["Blog", "Email"]
)

user_input = st.text_area(
    "Enter Topic / Purpose",
    height=120
)

tone = st.selectbox(
    "Select Tone",
    ["Professional", "Casual", "Friendly", "Technical", "Persuasive"]
)

if content_type == "Blog":
    length = st.selectbox(
        "Blog Length",
        ["500 words", "1000 words", "1500 words"]
    )

    prompt = f"""
    Generate a blog post.

    Topic: {user_input}
    Tone: {tone}
    Length: {length}

    Include:
    - SEO Friendly Title
    - Meta Description
    - Introduction
    - Headings and Subheadings
    - Conclusion
    """

else:
    prompt = f"""
    Generate a professional email.

    Purpose: {user_input}
    Tone: {tone}

    Include:
    - Subject Line
    - Greeting
    - Email Body
    - Closing
    """

if st.button("Generate"):
    with st.spinner("Generating..."):
        response = model.generate_content(prompt)

        st.subheader("Generated Content")
        st.write(response.text)

        st.download_button(
            label="Download",
            data=response.text,
            file_name=f"{content_type.lower()}.txt",
            mime="text/plain"
        )

!streamlit run app.py

# Commented out IPython magic to ensure Python compatibility.
# %%writefile app.py
# import streamlit as st
# import google.generativeai as genai
# import os
# 
# # Configure Gemini API
# # Check if running in Colab to use google.colab.userdata
# # Otherwise, try st.secrets or st.text_input
# API_KEY = ""
# try:
#     from google.colab import userdata
#     API_KEY = userdata.get('GEMINI_API_KEY')
# except ImportError:
#     # Not in Colab, try Streamlit secrets or prompt user
#     API_KEY = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY"))
# 
# if not API_KEY:
#     st.warning("GEMINI_API_KEY not found. Please enter it below or set it in Colab secrets / Streamlit secrets.toml")
#     API_KEY = st.text_input("Enter Gemini API Key", type="password", help="You can get your API key from https://makersuite.google.com/key")
#     if not API_KEY:
#         st.stop()
# 
# genai.configure(api_key=API_KEY)
# 
# model = genai.GenerativeModel("gemini-pro-latest") # Using gemini-pro-latest as identified earlier
# 
# st.set_page_config(page_title="AI Blog & Email Generator", layout="wide")
# 
# st.title("✍️ AI Blog & Email Generator")
# 
# content_type = st.radio(
#     "Select Content Type",
#     ["Blog", "Email"]
# )
# 
# user_input = st.text_area(
#     "Enter Topic / Purpose",
#     height=120
# )
# 
# tone = st.selectbox(
#     "Select Tone",
#     ["Professional", "Casual", "Friendly", "Technical", "Persuasive"]
# )
# 
# if content_type == "Blog":
#     length = st.selectbox(
#         "Blog Length",
#         ["500 words", "1000 words", "1500 words"]
#     )
# 
#     prompt = f"""
#     Generate a blog post.
# 
#     Topic: {user_input}
#     Tone: {tone}
#     Length: {length}
# 
#     Include:
#     - SEO Friendly Title
#     - Meta Description
#     - Introduction
#     - Headings and Subheadings
#     - Conclusion
#     """
# 
# else:
#     prompt = f"""
#     Generate a professional email.
# 
#     Purpose: {user_input}
#     Tone: {tone}
# 
#     Include:
#     - Subject Line
#     - Greeting
#     - Email Body
#     - Closing
#     """
# 
# if st.button("Generate"):
#     with st.spinner("Generating..."):
#         response = model.generate_content(prompt)
# 
#         st.subheader("Generated Content")
#         st.write(response.text)
# 
#         st.download_button(
#             label="Download",
#             data=response.text,
#             file_name=f"{content_type.lower()}.txt",
#             mime="text/plain"
#         )

!pip install google-generativeai

import google.generativeai as genai
from google.colab import userdata # Import userdata

# Configure Gemini API using the API key from Colab secrets
# Replace 'GEMINI_API_KEY' with the name you used in Colab secrets if it's different
API_KEY = userdata.get('GEMINI_API_KEY')

genai.configure(api_key=API_KEY)

# print("Listing available models:")
# for m in genai.list_models():
#     if "generateContent" in m.supported_generation_methods:
#         print(m.name)

model = genai.GenerativeModel("gemini-2.5-flash") # Using gemini-2.5-flash from the available list

prompt = "Write a 1000-word blog on Artificial Intelligence."

response = model.generate_content(prompt)

print(response.text)

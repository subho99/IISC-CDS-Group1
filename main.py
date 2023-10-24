import streamlit as st
from PIL import Image

st.header("Group1 - Capstone project")

import base64

@st.cache_data
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    body {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

set_png_as_page_bg('genAi.png')

# image = Image.open('parsa-mahmoudi-8x2iLiQ6J_U-unsplash.jpg')
# st.image(image, caption='IISc - Batch 5 - Capstone Group-1 project')

# st.write("Creator: User Neil Iris (@neil_ingham) from Unsplash")
# st.write("License: Do whatever you want https://unsplash.com/license")
# st.write("URL: https://unsplash.com/photos/I2UR7wEftf4")
    
# st.markdown(
#     """
#     <style>
#     .reportview-container {
#         background: url("https://www.example.com/image.jpg");
#     }
#    </style>
#     """,
#     unsafe_allow_html=True
# )

# page_bg_img = """
# <style>
# [data-testid ="stAppViewContainer"] {
# background-image: url("https://www.google.com/url?sa=i&url=https%3A%2F%2Funsplash.com%2Fphotos%2Fa-room-filled-with-lots-of-wooden-shelves-w5QDlbjJwEY&psig=AOvVaw1jOCXrmgxH6MwKugw4-Uj5&ust=1698151166097000&source=images&cd=vfe&ved=0CBEQjRxqFwoTCIDC2JyYjIIDFQAAAAAdAAAAABAE");
# background-size: cover;
# }
# </style>
# """
# st.markdown(page_bg_img, unsafe_allow_html=True)
#st.set_page_config(page_title="GAN based video generation", layout="wide")

# st.header("Group1 - Capstone project")

# st.markdown(
#     """
#     <link rel="stylesheet" type="text/css" href="https://www.example.com/style.css">
#     """,
#     unsafe_allow_html=True
# )

# st.markdown(
#     """
#     <style>
#     .reportview-container {
#          background: url("https://www.example.com/image.jpg");
#     }
#    </style>
#     """,
#     unsafe_allow_html=True
# )


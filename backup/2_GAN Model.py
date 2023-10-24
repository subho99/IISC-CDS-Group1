import streamlit as st
from pages.scripts.app2 import load_image, generate_video
from pages.scripts.content.util import get_classified_lable_file_path


st.header("GAN Model")

#New............................AK    
with open(get_classified_lable_file_path(), 'r') as f:
    classified_label = f.read()

st.write("The classified label:", classified_label)

user_input1 = st.text_input("Enter the batch size", help="The recommended size is 1 or 2 for testing", placeholder="1")
user_input2 = st.text_input("Enter the no. of epochs", help="The recommended size is 1 or 2 for testing", placeholder="1" )
user_input2 = st.text_input("Enter the activity", help="Name of the activity", placeholder="Archery" )

valid_inputs = True
batch_size = 0
num_epochs = 0
if st.button('Generate Video'):
    if len(user_input1) > 0: 
        batch_size = int(user_input1)
        if batch_size == 0 or  batch_size > 2:
            valid_inputs = False
            st.error("The recommended batch size is 1-2 only")
    else:
        valid_inputs = False
        st.error("Enter the batch size")

    if len(user_input2) > 0: 
        num_epochs = int(user_input2)
        if num_epochs == 0 or num_epochs > 2:
            valid_inputs = False
            st.error("The recommended epoch size is 1-2 only")
    else:
        valid_inputs = False
        st.error("Enter the epoch size")

    if valid_inputs == True:
        generate_video(batch_size, num_epochs)
        st.write("Video link")


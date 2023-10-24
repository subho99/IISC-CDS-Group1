import streamlit as st
import time

from pages.scripts.app import predict_resume_with_rf, predict_resume_with_xgb, predict_resume_with_lgb 
from pages.scripts.content.util import get_classified_lable_file_path
from pages.scripts.app2 import generate_video

progress_text = "Loading in progress. Please wait."
my_bar = st.progress(0, text=progress_text)

for percent_complete in range(100):
    time.sleep(0.01)
    my_bar.progress(percent_complete + 1, text=progress_text)
time.sleep(1)
my_bar.empty()


tabs_font_css = """
<style>
div[class*="stTextInput"] label p {
  font-size: 1rem;
  color: black;
}
</style>
"""
st.write(tabs_font_css, unsafe_allow_html=True)
st.header("Generate video using DCGAN")

st.subheader("Executing Text Classification")
with st.container():
    # Using the "with" syntax
    with st.form(key='my_form'):
        
        user_input = st.text_input("#1 Enter the Activity description", help="Example: A man playing archery in front of the garage in the morning", placeholder="A person playing Archery or A person playing football" )
        st.write("#2 Employ the model for classification.")
        model = st.radio(
        "select the model",
        ["RandomForest", "XGBoost", "Microsoft LGBM"],
        index=None,horizontal=True
        )
        button_clicked = False
        invalid_user_input = False
        model_selected = True
        classified_label = ""
        submit_button = st.form_submit_button(label='Classify')
        if submit_button:
            button_clicked = True
            if len(user_input) > 0: 
                if model == "RandomForest":
                    pred_action_label_rf = predict_resume_with_rf(user_input)
                    classified_label = pred_action_label_rf[0]
                elif model == "XGBoost":    
                    # Predicting the unknown action label
                    pred_action_label_xgb = predict_resume_with_xgb(user_input)
                    classified_label = pred_action_label_xgb[0]
                elif model == "Microsoft LGBM":  
                    # Predicting the unknown action label
                    pred_action_label_lgb = predict_resume_with_lgb(user_input)
                    classified_label = pred_action_label_lgb[0]
                else:
                    model_selected = False
            else:
                invalid_user_input = True
        if button_clicked == True:
            if invalid_user_input == True:
                st.error("Enter the activity")
            elif model_selected == False:
                st.error("Select the model")
            elif len(classified_label) > 0: 
                #st.success("Result : "+ classified_label)
                st.markdown(f'Result <font style="color:blue;font-size:15px;">{classified_label}</font>', unsafe_allow_html=True)
            else:
                st.warning("No matching action generated")
st.subheader("Executing the GAN Model")
with st.container():
    # Using the "with" syntax
    with st.form(key='my_form1'):           
        st.write("#3 Generate the video")
        #New............................AK    
        with open(get_classified_lable_file_path(), 'r') as f:
            classified_label = f.read()
        st.markdown(f'The classified text for generating the video: <font style="color:blue;font-size:15px;">{classified_label}</font>', unsafe_allow_html=True)

        with st.expander("Advanced options"):    
            custom_customized_label = st.text_input("Enter the activity", help="Name of the action like PlayingTabla or Football", placeholder="Enter the exact action class")
            show_images = st.checkbox("Show images")
        submit_button = st.form_submit_button(label='Generate Video')
        if submit_button:
            if len(custom_customized_label) > 0:
                generate_video(custom_customized_label, show_images)
            else:
                generate_video(classified_label, show_images)
            



# Streamlit Documentation: https://docs.streamlit.io/

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from streamlit_card import card # https://github.com/gamcoh/st-card
import base64
from st_aggrid import AgGrid # https://github.com/PablocFonseca/streamlit-aggrid
from st_vizzu import * # https://github.com/avrabyt/Streamlit-ipyvizzu



# Title/Text
st.title("Soldier Race Prediction")
st.text("Here you can predict a soldier race based on set of inputs.")


with open('army_pic.jpg', "rb") as f:
    data = f.read()
    encoded = base64.b64encode(data)
data = "data:image/png;base64," + encoded.decode("utf-8")


hasClicked = card(
  title="Original notebook",
  text="Here",
  image= data,
  url="https://github.com/0xArwa/soldier-race-prediction",
  on_click=lambda: print("clicked!")
)


st.subheader('Dataset Overview')

st.info(':bulb: NOTE: this is the cleaned df after data preproccesing to get a \
        better idea about the data the model was actually trained on.')
df = pd.read_csv("cleaned_dataset.csv", index_col=0)

st.dataframe(df.describe().T)  

st.text("statistical overview about the numerical columns.")
# --- First Vis ------ 

# Create ipyvizzu Object with the DataFrame
obj = create_vizzu_obj(df)

# Preset plot usage. Preset plots works directly with DataFrames.
bar_obj = bar_chart(df,
            x = "Age", 
            y = "DODRace",
            title= "Age vs. Soldiors Race"
            )

# Animate with defined arguments 
anim_obj = beta_vizzu_animate( bar_obj,
    x = "Gender",
    y =  ["DODRace", "Age"],
    title = "Age vs. Soldiors Race (Based on Gender)",
    label= "DODRace",
    color="Gender",
    legend="color",
    sort="byValue",
    reverse=True,
    align="center",
    split=False,
)

# Animate with general dict based arguments 
_dict = {"size": {"set": "DODRace"}, 
    "geometry": "circle",
    "coordSystem": "polar",
    "title": "General overview",
    }
anim_obj2 = vizzu_animate(anim_obj,_dict)


# Visualize within Streamlit
with st.container(): # Maintaining the aspect ratio
    st.button("Animate", key=1)
    vizzu_plot(anim_obj2)

# plot disc
st.text("The dataset is highly imblanced with the white race as the most frequent race in\n" 
        "the dataset and Hispanic race is the least frequent race.")

# ---- 2nd vis -----
obj2 = create_vizzu_obj(df)

bar_obj2 = bar_chart(df,
            x = "stature", 
            y = "SubjectsBirthLocation",
            title= "stature based on birth location",
            #height= "700px"
            )

anim_obj_ = beta_vizzu_animate( bar_obj2,
    x = "DODRace",
    y =  ["SubjectsBirthLocation", "stature"],
    title = "stature vs. birth location (Based on Race)",
    label= "SubjectsBirthLocation",
    color="DODRace",
    legend="color",
    sort="byValue",
    reverse=True,
    align="center",
    split=False,
)

_dict2 = {"size": {"set": "SubjectsBirthLocation"}, 
    "geometry": "circle",
    "coordSystem": "polar",
    "title": "General overview",
    }
anim_obj3 = vizzu_animate(anim_obj_,_dict2)

with st.container(): # Maintaining the aspect ratio
    st.button("Animate", key=2)
    vizzu_plot(anim_obj3)

# plot disc
st.text("The white race was mostly born inside the united states.")


# Markdown
st.markdown("Streamlit is **_really_ cool** :+1:")
st.markdown("# This is a markdown")
st.markdown("## This is a markdown")
st.markdown("### This is a markdown")

# Header/Subheader
st.header('This is a header')
st.subheader('This is a subheader')

# Success/Info/Error
st.success('This is a success message!')
st.info('This is a purely informational message')
st.error("This is an error.")
st.warning("This is a warning message!")
st.exception("NameError('name there is not defined')")

# Help
st.help(range)

# Write
st.write("Hello World! :sunglasses:")
st.write(range(10))

# Add image
#img = Image.open("images.jpeg")
#st.image(img, caption="cattie", width=300)

# Add video

#my_video = open("ml.mov",'rb')
#st.video(my_video)

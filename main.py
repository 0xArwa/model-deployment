# Streamlit Documentation: https://docs.streamlit.io/

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from streamlit_card import card # https://github.com/gamcoh/st-card
import base64
from st_vizzu import * # https://github.com/avrabyt/Streamlit-ipyvizzu



# Title/Text
st.title("Car Price Prediction")
st.text("Here you can predict a car price based on set of inputs.")


with open('car_pic.jpg', "rb") as f:
    data = f.read()
    encoded = base64.b64encode(data)
data = "data:image/png;base64," + encoded.decode("utf-8")


hasClicked = card(
  title="Autoscout Original Data",
  text="Here",
  image= data,
  url="https://www.autoscout24.com",
  on_click=lambda: print("clicked!")
)


st.subheader('Dataset Overview')

st.info(':bulb: NOTE: this is the cleaned df after data preproccesing to get a \
        better idea about the data the model was actually trained on.')
df = pd.read_csv("cleaned_dataset.csv")

st.dataframe(df.describe().T)  

st.text("Statistical overview about the numerical columns.")
# --- First Vis ------ 

# Create ipyvizzu Object with the DataFrame
obj = create_vizzu_obj(df)

# Preset plot usage. Preset plots works directly with DataFrames.
bar_obj = bar_chart(df,
            x = "price", 
            y = "location",
            title= "price vs. location Race",
            )

# Animate with defined arguments 
anim_obj = beta_vizzu_animate( bar_obj,
    x = "location",
    y =  ["type", "price"],
    title = "Car price based on location",
    label= "location",
    color="type",
    legend="color",
    sort="byValue",
    reverse=True,
    align="center",
    split=False,
)

# Animate with general dict based arguments 
_dict = {"size": {"set": "location"}, 
    "geometry": "circle",
    "coordSystem": "polar",
    "title": "General overview",
    }
anim_obj2 = vizzu_animate(anim_obj,_dict)


# Visualize within Streamlit
with st.container(): # Maintaining the aspect ratio
    st.button("Animate", key=1)
    st.caption(":point_up_2: please click this button")
    vizzu_plot(anim_obj2)


# plot disc
st.caption("- Overall, the plot shows that the average price of cars varies widely depending on the location.")

# ---- 2nd vis -----
obj2 = create_vizzu_obj(df)

bar_obj2 = bar_chart(df,
            x = "body_type", 
            y = "age",
            title = "Age of the car and the body type",
            #height= "700px"
            )

anim_obj_ = beta_vizzu_animate( bar_obj2,
    x = "body_type",
    y =  ["age", "power_kW"],
    title = "The car age vs. the power in KW",
    label= "age",
    color="body_type",
    legend="color",
    sort="byValue",
    reverse=True,
    align="center",
    split=False,
)

_dict2 = {"size": {"set": "age"}, 
    "geometry": "circle",
    "coordSystem": "polar",
    "title": "Car body type based on age and power in KW",
    }
anim_obj3 = vizzu_animate(anim_obj_,_dict2)

with st.container(): # Maintaining the aspect ratio
    st.button("Animate", key=2)
    st.caption(":point_up_2: please click this button")
    vizzu_plot(anim_obj3)



# plot disc
st.caption(" - Compact cars are the most common body type for cars with low power. This is likely because compact cars are more fuel-efficient and affordable than larger cars.")
st.caption(" - SUVs are the most common body type for cars with medium power. This is likely because SUVs offer more space and versatility than compact cars or sedans.")
st.caption(" - Luxury sedans are the most common body type for cars with high power. This is likely because luxury sedans offer the best performance and comfort.")



# delete this before production 
st.error("Delete this before production!!")
data = pd.DataFrame({"Column": df.columns, "Type": df.dtypes}).reset_index(drop=True)

st.table(data)


st.subheader('Car Specifications')

st.text('Please select the options of the car in order to predic its price')

#  ------------------- inputs --------------------------------

# selected_features = ['make_model', 'location', 'body_type','gearbox',
               #      'engine_size', 'co_emissions','drivetrain','empty_weight','energy_efficiency_class',
                #     'comfort_&_convenience_Package','age', 'power_kW', 'safety_&_security_Package']



# make_model
st.markdown("#### Select the car model")
make_model = st.selectbox('', options = df.make_model.unique().tolist())


# location
st.markdown("#### Select the location")
location = st.selectbox('', options=df.location.unique().tolist())

# body type
st.markdown("#### Select the car body type")
body_type = st.radio('', options=df.body_type.unique().tolist())


# gearbox
st.markdown("#### Select the gearbox")
gearbox = st.radio('', options=df.gearbox.unique().tolist())

# engine_size
st.markdown("#### Engine size")
engine_size = st.slider("",min_value = float(df.engine_size.min()), max_value = float(df.engine_size.max()))

# co_emissions
st.markdown("#### Co emissions")
co_emissions = st.slider("",min_value = float(df.co_emissions.min()), max_value = float(df.co_emissions.max()))

# drivetrain
st.markdown("#### Drivetrain")
drivetrain = st.radio('', options=df.drivetrain.unique().tolist())

# empty_weight
st.markdown("#### Car empty weight")
empty_weight = st.slider("",min_value = float(df.empty_weight.min()), max_value = float(df.empty_weight.max()))

# age
st.markdown("#### Car age")
age = st.number_input("", min_value = int(df.age.min()), max_value = int(df.age.max()), step=1)

# power_kW
st.markdown("#### Car power in KW")
power_kW = st.slider("",min_value = float(df.power_kW.min()), max_value = float(df.power_kW.max()))

# energy_efficiency_class
st.markdown("#### Select energy efficiency class")
energy_efficiency_class = st.selectbox('', options=df.energy_efficiency_class.unique().tolist())

# energy_efficiency_class
st.markdown("#### Select comfort & convenience package")
comfort_convenience_package = st.selectbox('', options=df['comfort_&_convenience_Package'].unique().tolist())

# energy_efficiency_class
st.markdown("#### Select safety security package")
safety_security_package = st.selectbox('', options=df['safety_&_security_Package'].unique().tolist())

import pickle
filename = "GB_regressor_model.sav"
model=pickle.load(open(filename, "rb"))

my_dict = {'make_model':make_model,
           'location':location,
           'body_type':body_type,
           'gearbox':gearbox,
           'engine_size':engine_size,
           'co_emissions':co_emissions,
           'drivetrain':drivetrain,
           'empty_weight':empty_weight,
           'energy_efficiency_class':energy_efficiency_class,
            'comfort_&_convenience_Package':comfort_convenience_package,
            'age':age, 
            'power_kW':power_kW, 
            'safety_&_security_Package':safety_security_package
           }

df = pd.DataFrame.from_dict([my_dict])
st.table(df)

# Prediction with user inputs
predict = st.button("Predict")
result = model.predict(df)
if predict :
    st.text('Predicted car price is: ')
    st.success(result[0].__round__(3))

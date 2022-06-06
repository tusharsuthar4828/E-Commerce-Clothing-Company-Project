

import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.ensemble import GradientBoostingRegressor


st.title('Ecommerce_Clothing_Predictions')
st.sidebar.header('Input_Features_Variables')

image = Image.open('b.jpg')
st.image(image,'')


def input_var():
    avg_sess_len = st.sidebar.number_input('Avg session length')
    tim_app = st.sidebar.number_input('Time on App')
    tim_web = st.sidebar.number_input('Time on Website')
    len_mem = st.sidebar.number_input('Length of MemberShip')
    data = {'AVG_SESS_LEN':avg_sess_len,
            'TIM_APP':tim_app,
            'TIM_WEB':tim_web,
            'LEN_MEM':len_mem}
    features = pd.DataFrame(data,index=[0])
    return features

ecom_df = input_var()
st.header('Model_Deployment : Gradient_Boosting_Regressor')
st.subheader('Input_Features_Variables')
st.write(ecom_df)

cus_data = pd.read_csv('Ecommerce.csv')
cus_data.drop(['Customer ID'],inplace=True,axis=1)


X = cus_data.drop('Yealy amount spent', axis=1)
y = cus_data['Yealy amount spent']


gb_model = GradientBoostingRegressor(random_state=42, learning_rate=0.01)
gb_model.fit(X,y)

gb_model_pred = gb_model.predict(ecom_df)
if st.button('Predict'):
   st.header('Predicting the value of GB_MODEL : {}'.format(int(gb_model_pred)))





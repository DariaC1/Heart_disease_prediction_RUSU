import streamlit as st
import pandas as pd

# naziv temeljen na nazivu dataseta
st.title("Cardiovascular disease prediction")

st.write('')

st.subheader("Data information: ")

st.write("Systolic blood pressure is higher number.")
st.write("Diastolic blood pressure is lower number.")
st.write("Cholesterol levels are: 1 - normal, 2 - above normal, 3 - well above normal")
st.write("Glucose levels are: 1 - normal, 2 - above normal, 3 - well above normal")

st.sidebar.markdown("## Choose atributes")

def get_user_input():
    
    numbers = [False]
    features = 0 # nužno postaviti da bi radilo dok nije odabran ni jedan atribut
    data = False
    
    col_1 = st.sidebar.checkbox('Select age')
    if col_1:
        age = st.sidebar.slider('20 - 90', 20, 90, 55, key = 1)
        age = age * 365
        data = {'age' : age}
        features = pd.DataFrame(data, index = [0])
        numbers = [True]
       
    col_2 = st.sidebar.checkbox('Select sex')
    if col_2: 
        sex = st.sidebar.slider('1 - female, 2 - male', 1, 2, 1, key = 2)
        if data:
            features['sex'] = sex
            numbers.append(True)
        else:
            data = {'sex' : sex}
            features = pd.DataFrame(data, index = [0])
    else: 
        numbers.append(False)
        
    col_3 = st.sidebar.checkbox('Select height')
    if col_3:
        height = st.sidebar.slider('in cm', 100, 210, 155, key = 3)
        if data:
            features['height'] = height
            numbers.append(True)
        else:
            data = {'height' : height}
            features = pd.DataFrame(data, index = [0])
    else: 
        numbers.append(False)
  
    col_4 = st.sidebar.checkbox('Select weight')
    if col_4:
        weight = st.sidebar.slider('in kg', 30, 200, 115, key = 4)
        if data:
            features['weight'] = weight
            numbers.append(True)
        else:
            data = {'weight' : weight}
            features = pd.DataFrame(data, index = [0])
    else: 
        numbers.append(False)
        
    col_5 = st.sidebar.checkbox('Select systolic blood pressure')
    if col_5:
        ap_hi = st.sidebar.slider(' ', 80, 180, 130, key = 5)
        if data:
            features['ap_hi'] = ap_hi
            numbers.append(True)
        else:
            data = {'ap_hi' : ap_hi}
            features = pd.DataFrame(data, index = [0])
    else: 
        numbers.append(False)

    col_6 = st.sidebar.checkbox('Select diastolic blood pressure')
    if col_6:
        ap_lo = st.sidebar.slider(' ', 40, 140, 90, key = 6)
        if data:
            features['ap_lo'] = ap_lo
            numbers.append(True)
        else:
            data = {'ap_lo' : ap_lo}
            features = pd.DataFrame(data, index = [0])
    else: 
        numbers.append(False)
        
    col_7 = st.sidebar.checkbox('Select cholesterol')
    if col_7:
        cholesterol = st.sidebar.slider('levels(3): 1 to 3', 1, 3, 2, key = 7)
        if data:
            features['cholesterol'] = cholesterol
            numbers.append(True)
        else:
            data = {'cholesterol' : cholesterol}
            features = pd.DataFrame(data, index = [0])
    else: 
        numbers.append(False)
        
    col_8 = st.sidebar.checkbox('Select glucose levels')
    if col_8:
        gluc = st.sidebar.slider('levels(3): 1 to 3', 1, 3, 2, key = 8)
        if data:
            features['gluc'] = gluc
            numbers.append(True)
        else:
            data = {'gluc' : gluc}
            features = pd.DataFrame(data, index = [0])
    else: 
        numbers.append(False)
        
    col_9 = st.sidebar.checkbox('Select smoking')
    if col_9:
        smoke = st.sidebar.slider('0 - not smoking, 1 - smoking', 0, 1, 0, key = 9)
        if data:
            features['smoke'] = smoke
            numbers.append(True)
        else:
            data = {'smoke' : smoke}
            features = pd.DataFrame(data, index = [0])
    else: 
        numbers.append(False)
        
    col_10 = st.sidebar.checkbox('Select alcohol drinking')
    if col_10:
        alco = st.sidebar.slider('0 - not drinking, 1 - drinking', 0, 1, 0, key = 10)
        if data:
            features['alco'] = alco
            numbers.append(True)
        else:
            data = {'alco' : alco}
            features = pd.DataFrame(data, index = [0])
    else: 
        numbers.append(False)
        
    col_11 = st.sidebar.checkbox('Select physical activity')
    if col_11:
        active = st.sidebar.slider('0 - no, 1 - yes', 0, 1, 0, key = 11)
        if data:
            features['active'] = active
            numbers.append(True)
        else:
            data = {'active' : active}
            features = pd.DataFrame(data, index = [0])
    else: 
        numbers.append(False)
             
    numbers.append(False)
    return features, numbers


user_input, numbers = get_user_input()

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

st.write('')

option = st.selectbox("Choose clasification algorithm", ("Logistic Regression", "Random Forest", "k-Nearest Neighbors", "Gradient Boosting"))
button = st.button("Get prediction")

def RFC_fun(X_train, X_test, Y_train, Y_test):
      RFC = RandomForestClassifier(n_estimators = 100, max_features = "log2")
      RFC.fit(X_train, Y_train)
      st.write(str(accuracy_score(Y_test, RFC.predict(X_test)) * 100) + '%')
      prediction = RFC.predict(user_input)
      return prediction

def LR_fun(X_train, X_test, Y_train, Y_test):
      LR = LogisticRegression()
      LR.fit(X_train, Y_train)
      st.write(str(accuracy_score(Y_test, LR.predict(X_test)) * 100) + '%')
      prediction = LR.predict(user_input)
      return prediction
    
def kN_fun(X_train, X_test, Y_train, Y_test):
    kN = KNeighborsClassifier(n_neighbors = 8)
    kN.fit(X_train, Y_train)
    st.write(str(accuracy_score(Y_test, kN.predict(X_test)) * 100) + '%')
    prediction = kN.predict(user_input)
    return prediction  

def GBC_fun(X_train, X_test, Y_train, Y_test):
    Grad_Boosting = GradientBoostingClassifier( n_estimators = 100, max_depth = 1, learning_rate = 1,
                                           random_state = 0)
    Grad_Boosting.fit(X_train, Y_train) 
    st.write(str(accuracy_score(Y_test, Grad_Boosting.predict(X_test)) * 100) + '%')
    prediction = Grad_Boosting.predict(user_input) 
    return prediction
   
if button:
    # pošto je velik dataset treba vremena da učita i stoga je bolje da se nalazi tu
    # da se ne bi učitavalo svaki put kada bi se promijenila lista atributa i time znatno
    # usporio program 
    heart_df = pd.read_csv ("cardio_train.csv", index_col = 0 )
    
    X = heart_df.iloc[:, numbers]
    Y = heart_df.iloc[:, -1]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)
   
    st.subheader('User Input:')
    st.write(user_input)
    st.subheader('Model Test Accuracy Score:')
   
    if (option == "Random Forest"):
      prediction = RFC_fun(X_train, X_test, Y_train, Y_test)
    elif (option == "Logistic Regression"):
        prediction = LR_fun(X_train, X_test, Y_train, Y_test)
    elif (option == "k-Nearest Neighbors"):
        prediction = kN_fun(X_train, X_test, Y_train, Y_test)
    elif (option == "Gradient Boosting"):
        prediction = GBC_fun(X_train, X_test, Y_train, Y_test)

    st.subheader('Classification result: ')
    st.write('')
    if (prediction == 0):
        st.markdown("**Predicted that there is no cardiovascular disease.**")
    else:
        st.markdown("**Predicted that cardiovascular disease is present.**")
    
    st.write(prediction)   

#streamlit run "D:\4.godina\Projekti\rusu\model_heart_v2.py"




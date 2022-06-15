import streamlit as st
st.set_page_config(page_title="Churn Dashboard",
                   page_icon=":bar_chart:", layout="wide")

#import plotly.figure_factory as ff
import plotly.graph_objects as go


import requests
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu

import pandas as pd
import plotly.express as px
import joblib

import database as db
import bcrypt

 
headerSection = st.container()
mainSection = st.container()
loginSection = st.container()
logOutSection = st.container()

def login(username, password):
    res = {}
    res['status'] = False
    res['msg'] = ""

    user = db.get_user(username)

    if(user == None):
        res['msg'] = "Invalid username and password"
    elif(bcrypt.checkpw(password.encode('utf-8'), user['password'].encode('utf-8'))):
        res['status'] = True
        st.session_state['user name'] = user['name']
        st.experimental_set_query_params(
            loggedIn=True,
            user= user['name']
        )
    else:
        res['msg'] = "Invalid password"

    
    return res
 
def LoggedOut_Clicked():
    st.session_state['loggedIn'] = False
    st.session_state['user name'] = ""
    st.experimental_set_query_params(
            loggedIn=False,
    )
    
def show_logout_page():
    loginSection.empty();
    st.button ("Log Out", key="logout", on_click=LoggedOut_Clicked)
    
def LoggedIn_Clicked(userName, password):
    if(userName == "" or password == ""):
        st.session_state['loggedIn'] = False;
        st.warning("Username and password is required")
    
    res = login(userName, password)

    if(res['status']):
        st.session_state['loggedIn'] = True
        # st.experimental_set_query_params(
        #     loggedIn=True
        # )
    else:
        st.session_state['loggedIn'] = False;
        st.error(res['msg'])
    
def show_login_page():
    with loginSection:
        if st.session_state['loggedIn'] == False:
            userName = st.text_input (label="Username", value="", placeholder="Enter your user name")
            password = st.text_input (label="Password", value="",placeholder="Enter password", type="password")
            st.button ("Login", on_click=LoggedIn_Clicked, args= (userName, password))

@st.cache
def get_data_from_csv():
    df = pd.read_csv('data/Train_preprocessed.csv')
    df = df.drop(columns=['Unnamed: 0'] , errors='ignore')
    df['voice_mail_plan_en'] = df.voice_mail_plan.map(dict(yes=1, no=0))
    df['intertiol_plan_en'] = df.intertiol_plan.map(dict(yes=1, no=0))
    df['total_plans'] = df['voice_mail_plan_en'] + df['intertiol_plan_en']
    return df


def get_model(file_path):
    loaded_model = joblib.load(file_path)
    return loaded_model


def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def get_filtered_df(args , df):
    col = [
             'total_calls',
             'total_minutes',
             'total_charges',
             ]
    df_filtered = df.copy()
    df_filtered = df_filtered.drop(columns=col , errors='ignore')
    df_filtered['total_calls'] = 0
    df_filtered['total_minutes'] = 0
    df_filtered['total_charges'] = 0
    if("day" in args):
        df_filtered['total_calls'] += df['total_day_min']
        df_filtered['total_minutes'] += df['total_day_calls']
        df_filtered['total_charges'] += df['total_day_charge']
    if("evening" in args):
        df_filtered['total_calls'] += df['total_eve_min']
        df_filtered['total_minutes'] += df['total_eve_calls']
        df_filtered['total_charges'] += df['total_eve_charge']
    if("night" in args):
        df_filtered['total_calls'] += df['total_night_minutes']
        df_filtered['total_minutes'] += df['total_night_calls']
        df_filtered['total_charges'] += df['total_night_charge']
    if("international" in args):
        df_filtered['total_calls'] += df['total_intl_minutes']
        df_filtered['total_minutes'] += df['total_intl_calls']
        df_filtered['total_charges'] += df['total_intl_charge']

    return df_filtered

@st.cache
def get_data_from_csv_model():
    df = pd.read_csv('data/model_data.csv')
    return df

def feature_importance(X, model):
    importances = model.feature_importances_
    # st.write(X.columns)
    fig = px.bar(x=X.columns, y=importances)
    return fig

def get_Table(p):
    df1 = p.describe().reset_index()
    header = df1.columns
    fig = go.Figure(data = go.Table(header = dict(values = list(header) , fill_color = "#002080" , align = 'center') ,
                                    cells = dict(values = df1 , fill_color = "#b3b3b3" , align = 'left')))
    #fig.update_layout(margin = dict(l=5 , r=5 , b=10 , t=10) , paper_bgcolor = "#000000")
    return fig

def show_correlations(dataframe, show_chart = True):
    corr = dataframe.corr()
    if show_chart == True:
        fig = px.imshow(corr,
                        text_auto=True,
                        width=900,
                        height=800)
    return fig
      

def show_dashboard():

    df = get_data_from_csv()
    df_new = get_data_from_csv_model()
    model_xg = get_model('data/model_xg.sav')
    model_cat = get_model('data/model_cat.sav')


    df['voice_mail_plan_en'] = df.voice_mail_plan.map(dict(yes=1, no=0))
    df['intertiol_plan_en'] = df.intertiol_plan.map(dict(yes=1, no=0))
    df['total_plans'] = df['voice_mail_plan_en'] + df['intertiol_plan_en']

    with mainSection:

        # ---- SIDEBAR ----

        with st.sidebar:

            show_logout_page()

            if('user name' in st.session_state and st.session_state['user name']!= ""):
                st.title('Welcome ' + st.session_state['user name'])     

            try:
                lottie_profile = load_lottieurl(
                    "https://assets3.lottiefiles.com/packages/lf20_hcjfe9tg.json")
                st_lottie(
                    lottie_profile,
                    speed=1,
                    reverse=False,
                    loop=True,
                    quality="low",  # medium ; high
                    height=None,
                    width=None,
                    key=None,
                )
            except Exception as e:
                print(e)

        # ---- NAVE BAR ----
        selected = option_menu(
            menu_title=None,  # required
            options=["Home", "Summarizer", "Predict"],  # required
            icons=["house", "book", "activity"],  # optional
            menu_icon="cast",  # optional
            default_index=0,  # optional
            orientation="horizontal",
            styles={
                "container": {"padding": "0!important", "background-color": "black"},
                "icon": {"color": "blue", "font-size": "25px"},
                "nav-link": {
                        "font-size": "25px",
                        "text-align": "left",
                        "margin": "0px",
                        "--hover-color": "#a6a6a6",
                },
                "nav-link-selected": {"background-color": "#666666"},
            },
        )

        if selected == "Home":
            churn = st.multiselect(
                'Churn Rate',
                df["Churn"].unique(),
                default=df["Churn"].unique()
            )

            location = st.multiselect(
                'Location',
                df["location_code"].unique(),
                default=df["location_code"].unique()
            )

            type = st.multiselect(
                'Type',
                ['day' , "evening" , "night" , "international"],
                default=['day' , "evening" , "night" , "international"]
            )

            df_selection = df.query(
                "Churn == @churn & location_code ==@location"
            )

            df_selection = get_filtered_df(type , df_selection)

            st.markdown("""---""")

            fig_call_1 = px.ecdf(df_selection, x="total_calls", color="Churn" , labels={"total_calls": "Total Calls"})
            fig_call_1.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=(dict(showgrid=False)))

            fig_min_1 = px.ecdf(df_selection, x="total_minutes", color="Churn" , labels={"total_minutes": "Total Minutes"})
            fig_min_1.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=(dict(showgrid=False)))

            fig_charge_1 = px.ecdf(df_selection, x="total_charges", color="Churn" , labels={"total_charges": "Total Charges"})
            fig_charge_1.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=(dict(showgrid=False)))

            fig_call_2 = px.scatter(df_selection, x="total_calls", y="customer_id", color="location_code",log_x= True, labels={"total_calls": "Total Calls" , "customer_id": "Customer Count"})
            fig_call_2.update_layout(
            xaxis=dict(tickmode="linear"),
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=(dict(showgrid=False)),)

            fig_min_2 = px.scatter(df_selection, x="total_minutes", y="customer_id",  color="location_code",log_x=True , labels={"total_minutes": "Total Minutes" , "customer_id": "Customer Count"})
            fig_min_2.update_layout(
            xaxis=dict(tickmode="linear"),
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=(dict(showgrid=False)),)

            fig_charge_2 = px.scatter(df_selection, x="total_charges", y="customer_id",  color="location_code", log_x=True , labels={"total_charges": "Total Charges" , "customer_id": "Customer Count"})
            fig_charge_2.update_layout(
            xaxis=dict(tickmode="linear"),
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=(dict(showgrid=False)),)

            fig_int_serve = px.bar(df_selection, x="intertiol_plan" , y=df_selection.index , color="Churn" ,  labels={"intertiol_plan": "International Services Plan" , "index": "Customer Count"})
            fig_int_serve.update_layout(
            xaxis=dict(tickmode="linear"),
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=(dict(showgrid=False)),)

            fig_voice_serve = px.bar(df_selection, x="voice_mail_plan" , y=df_selection.index , color="Churn" , labels={"voice_mail_plan": "Voice Mails Services Plan" , "index": "Customer Count"})
            fig_voice_serve.update_layout(
            xaxis=dict(tickmode="linear"),
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=(dict(showgrid=False)),)

            with st.container():
                st.subheader("Total Calls")
                left_column, right_column = st.columns(2)
                left_column.plotly_chart(fig_call_1, use_container_width=True)
                right_column.plotly_chart(fig_call_2, use_container_width=True)
            
            st.markdown("""---""")

            with st.container():
                st.subheader("Total Minutes")
                left_column, right_column = st.columns(2)
                left_column.plotly_chart(fig_min_1, use_container_width=True)
                right_column.plotly_chart(fig_min_2, use_container_width=True)
            
            st.markdown("""---""")

            with st.container():
                st.subheader("Total Charges")
                left_column, right_column = st.columns(2)
                left_column.plotly_chart(fig_charge_1, use_container_width=True)
                right_column.plotly_chart(fig_charge_2, use_container_width=True)
            
            st.markdown("""---""")

            with st.container():
                st.markdown("<h4 style='text-align: center; color: yellow;'>Services & Plans</h4><br/><br/>",
                unsafe_allow_html=True)

                left_column, right_column = st.columns(2)

                left_column.write("International Services")
                left_column.plotly_chart(fig_int_serve, use_container_width=True)

                right_column.write("Voice Services")
                right_column.plotly_chart(fig_voice_serve, use_container_width=True)

            st.markdown("""---""")

        if selected == "Summarizer":

            fig_churn = px.pie(df, values=df.index, names='Churn')
            fig_churn.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            )

            fig_plans = px.pie(df, values=df.index, names='total_plans')
            fig_plans.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            )

            #fig_heat = px.imshow(df, text_auto=True, aspect="auto")

            fig_heat = show_correlations(df_new)

            fig_importance = feature_importance(df_new, model_xg)
            fig_importance.update_layout(
            xaxis=dict(tickmode="linear"),
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=(dict(showgrid=False)),)
            

            fig_cus_serve = px.ecdf(df, x="customer_service_calls" , y=df.index , color="Churn" ,  labels={"customer_service_calls": "customer service calls" , "index": "Customer Count"})
            fig_cus_serve.update_layout(
            xaxis=dict(tickmode="linear"),
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=(dict(showgrid=False)),)

            #st.markdown("""---""")

            with st.expander("Data Description"):
                st.write(df.describe())

            st.markdown("""---""")

            with st.container():

                left_column, middle_column, right_column = st.columns(3)

                left_column.markdown("<h3 style='text-align: center; color: yellow;'>Churn Rate</h3>",
                unsafe_allow_html=True)
                left_column.plotly_chart(fig_churn, use_container_width=True)

                middle_column.markdown("<h3 style='text-align: center; color: yellow;'>Customer Services Calls</h3>",
                unsafe_allow_html=True)
                middle_column.plotly_chart(fig_cus_serve, use_container_width=True)

                right_column.markdown("<h3 style='text-align: center; color: yellow;'>Subcribed Plans</h3>",
                unsafe_allow_html=True)
                right_column.plotly_chart(fig_plans, use_container_width=True)
            
            st.markdown("""---""")

            with st.container():
                st.markdown("<h3 style='text-align: center; color: yellow;'>Data Correlations</h3><br/>",
                unsafe_allow_html=True)

                st.plotly_chart(fig_heat , use_container_width=True)

            st.markdown("""---""")

            with st.container():

                st.markdown("<h3 style='text-align: center; color: yellow;'>Feature Importance</h3>",
                unsafe_allow_html=True)   
                st.write("See feature importance using XGBooster method:")
                st.plotly_chart(fig_importance, use_container_width=True)

            st.markdown("""---""")

        if selected == "Predict":

            with st.form("my_form"):
                st.write("Please fill out all the fields for new customer")

                # st.markdown("<div class='tooltip'>read!<span class='tooltiptext'>Please fill out all the fields for new customer</span></div>" , unsafe_allow_html=True)
            
                col1, col2, col3, col4 = st.columns(4)
                acc_length = col1.number_input("Account length: ")
                location_code = col2.selectbox(
                                'Location Code: ',
                                df.location_code.unique()
                            )  
                
                intertiol_plan = col3.selectbox(
                                'International plan: ',
                                df.intertiol_plan.unique()
                            )   
                
                voice_mail_plan = col4.selectbox(
                                'Voice Mail Plan: ',
                                df.voice_mail_plan.unique()
                            )   
                
                number_vm_messages = col1.number_input("VM message count: ")
                total_day_min = col2.number_input("Day minutes: ")
                total_day_calls = col3.number_input("Day calls: ")
                total_day_charge = col4.number_input("Day charge: ")
                total_eve_min = col1.number_input("Evening minutes: ")
                total_eve_calls = col2.number_input("Evening calls: ")
                total_eve_charge = col3.number_input("Evening charge: ")
                total_night_minutes = col4.number_input("Night minutes: ")
                total_night_calls = col1.number_input("Night calls: ")
                total_night_charge = col2.number_input("Night charge: ")
                total_intl_minutes = col3.number_input("International minutes: ")
                total_intl_calls = col4.number_input("International calls: ")
                total_intl_charge = col1.number_input("International charge: ")
                customer_service_calls = col2.number_input("Customer Service Calls: ")
                total_calls = total_day_calls + total_eve_calls + total_night_calls + total_intl_calls
                total_charge = total_day_charge + total_eve_charge + total_night_charge + total_intl_charge
                total_mins = total_day_min + total_eve_min + total_night_minutes + total_intl_minutes
                
                location_code_445 = 0
                location_code_452 = 0
                location_code_547 = 0
                
                if location_code == '445':
                    location_code_445 = 1
                elif location_code == '452':
                    location_code_452 = 1
                else:
                    location_code_547 = 1
                
                if intertiol_plan == 'yes':
                    intertiol_plan = 1
                else:
                    intertiol_plan = 0
                    
                if voice_mail_plan == "yes":
                    voice_mail_plan = 1
                else:
                    voice_mail_plan = 0

                no_of_plans = intertiol_plan + voice_mail_plan
                
            # Every form must have a submit button.
                try:
                    submitted = st.form_submit_button("Submit")
                    if submitted:
                        new_row = [acc_length,
                                intertiol_plan,
                                voice_mail_plan,
                                number_vm_messages,
                                total_day_min,
                                total_day_calls,
                                total_day_charge,
                                total_eve_min,
                                total_eve_calls,
                                total_eve_charge,
                                total_night_minutes,
                                total_night_calls,
                                total_night_charge,
                                total_intl_minutes,
                                total_intl_calls,
                                total_intl_charge,
                                customer_service_calls,
                                location_code_452,
                                location_code_445,
                                location_code_547,
                                total_charge,
                                total_calls,
                                total_mins,
                                no_of_plans
                                ]
                        X_columns = ['account_length', 'intertiol_plan', 'voice_mail_plan',
                                    'number_vm_messages', 'total_day_min', 'total_day_calls',
                                    'total_day_charge', 'total_eve_min', 'total_eve_calls',
                                    'total_eve_charge', 'total_night_minutes', 'total_night_calls',
                                    'total_night_charge', 'total_intl_minutes', 'total_intl_calls',
                                    'total_intl_charge', 'customer_service_calls', 'location_code_452',
                                    'location_code_445', 'location_code_547', 'total_charge', 'total_calls',
                                    'total_min', 'no_of_plans']
                        
                        col1, col2, col3 = st.columns(3)
                        
                        X = pd.DataFrame (new_row).T
                        X.columns = X_columns 
                        
                        # XGBooster
                        col1.markdown("""""")
                        col1.subheader("XGBooster")
                        xg_prediction = model_xg.predict(X)
                        if xg_prediction == 0:
                            churn_xg = "NO CHURN"
                        else:
                            churn_xg = "CHURN"
                        
                        col1.write(str(churn_xg))
                        

                        
                        
                        # Cat Boost
                        col2.markdown("""""")
                        col2.subheader("Cat Booster")
                        cat_prediction = model_cat.predict(X)
                        if cat_prediction == 0:
                            churn_cat = "NO CHURN"
                        else:
                            churn_cat = "CHURN"
                        
                        col2.write(str(churn_cat))
                        
                        
        
                except TypeError as err:
                    st.write('err', err)


with headerSection:
    st.markdown("<h1 style='text-align: center; color: red;'><span> 📊 </span>Customer Churn Dashboard</h1>",
            unsafe_allow_html=True)

    st.markdown("##")
    #first run will have nothing in session_state
    if 'loggedIn' not in st.session_state:
        st.session_state['loggedIn'] = None
    if st.session_state['loggedIn'] is None:
        try:
            params = st.experimental_get_query_params()
            #print(params)
            if('loggedIn' in params) and params['loggedIn'][0]:
                st.session_state['loggedIn'] = True
                if('user' in params): st.session_state['user name'] = params['user'][0]
                show_dashboard() 
            else:
                st.session_state['loggedIn'] = False
                show_login_page()
        except Exception as e:
            print(e)
            st.session_state['loggedIn'] = False
            show_login_page()
    else:
        print(st.session_state['loggedIn'])
        if st.session_state['loggedIn']:   
            show_dashboard()  
        else:
            show_login_page()

 # ---- HIDE STREAMLIT STYLE ----
hide_st_style = """
                    <style>
                    #MainMenu {visibility: hidden;}
                    footer {visibility: hidden;}
                    header {visibility: hidden;}
                    </style>
                    """
st.markdown(hide_st_style, unsafe_allow_html=True)

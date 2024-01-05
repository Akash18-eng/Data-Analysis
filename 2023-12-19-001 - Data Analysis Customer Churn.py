#!/usr/bin/env python
# coding: utf-8

# # 1. Loading libraries and data

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


# In[2]:


df = pd.read_csv('customer_churn.csv')


# # 2. Interpreting the dataset

# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.isnull().sum()


# In[7]:


df.dtypes


# # 3. Data cleaning

# In[8]:


df = df.drop(columns = ['customerID'], axis = 1)
df.head()


# In[9]:


df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors = 'coerce')


# In[10]:


df.isnull().sum()


# In[11]:


df.fillna(df['TotalCharges'].mean())


# In[12]:


df.drop(labels = df[df['tenure'] == 0].index, axis = 0 , inplace = True)


# In[13]:


df[df['tenure'] == 0].index


# In[14]:


df['SeniorCitizen'] = df['SeniorCitizen'].map({0: "No", 1: "Yes"})


# In[15]:


df.head()


# In[16]:


df.describe()


# In[17]:


df.describe(include = 'O')


# In[18]:


df.describe(include = 'O').T


# In[19]:


df.describe(include = 'all')


# In[20]:


df


# In[21]:


df1 = df
df1['Churn'].replace(to_replace = 'Yes', value = 1, inplace = True)
df1['Churn'].replace(to_replace = 'No', value = 0, inplace = True)
df_dummies = pd.get_dummies(df1)
df_dummies.head()


# In[22]:


plt.figure(figsize = (15, 8))
sns.set(style = 'white')
df_dummies.corr()['Churn'].sort_values(ascending = False).plot(kind = 'bar')


# In[23]:


get_ipython().system('pip install sweetviz')


# In[24]:


import sweetviz as sv
my_report = sv.analyze(df)
my_report.show_html('report_html')


# # 4. Visualizing the data

# In[25]:


gender_labels = ['Male', 'Female']
churn_labels = ['No', 'Yes']

fig = make_subplots(rows = 1, cols = 2, specs = [[{'type':'domain'}, {'type':'domain'}]])
fig.add_trace(go.Pie(labels = gender_labels, values = df['gender'].value_counts(), name = 'Gender'), 1, 1)
fig.add_trace(go.Pie(labels = churn_labels, values = df['Churn'].value_counts(), name = 'Churn'), 1, 2)

fig.update_traces(hole = .55, hoverinfo = "label+percent+name", textfont_size = 16)

fig.update_layout(title_text = "Gender and Churn Distributions",
                 annotations = [dict(text = 'Gender', x=0.16, y=0.5, font_size=20, showarrow=False),
                               dict(text = 'Gender', x=0.83, y=0.5, font_size=20, showarrow=False)])

fig.data[0].marker.colors = ('#7fcdff', '#326ada')
fig.data[1].marker.colors = ('#56c175', '#ff9b35')
fig.show()


# In[26]:


#color_discrete_map = {"Month-to-month": "#7fcdff", "One year": "#326ada", "Two year": "#ff9b355"}
fig = px.histogram(df, x = 'Churn', color = 'Contract', barmode = 'group')
fig.update_layout(width = 700, height = 500, bargap = 0.1)
fig.show()

75% of customer who have Month-to-Month Contract have opted to move out as compared to 13% of customers who have signed One Year Contract and 3% of customers who have signed Two Year Contract.
# In[27]:


fig = px.histogram(df, x = 'Churn', color = "PaymentMethod", title = "<b>Churn distribution w.r.t Custome Payment Method</b>", text_auto = True)

fig.update_layout(width = 700, height = 500, bargap= 0.1)
fig.data[0].marker.color = ('#7fcdff')
fig.data[1].marker.color = ('#ff9b35')
fig.data[2].marker.color = ('#56c175')
fig.show()

Majority of the customers who moved out were having Electronic Check as Payment Method and others who opted for Credit-Card automatic transfer / Bank Automatic Transfer and Mailed Check as Payment Method were less likely to switch.
# In[28]:


df[df['gender'] == "Male"][["InternetService", "Churn"]].value_counts()


# In[29]:


df[df['gender'] == "Female"][["InternetService", "Churn"]].value_counts()


# In[30]:


fig = go.Figure()

colors = {'Female': 'steelblue', 'Male': 'firebrick'}

fig.add_trace(go.Bar(
    x = [['Churn:No', 'Churn:No', 'Churn:Yes', 'Churn:Yes'],
        ["Female", "Male", "Female", "Male"]],
    y = [965, 992, 219, 240],
    name = 'DSL'))
fig.add_trace(go.Bar(
    x = [['Churn:No', 'Churn:No', 'Churn:Yes', 'Churn:Yes'],
        ['Female', 'Male', 'Female', 'Male']],
    y = [889, 910, 664, 663],
    name = 'Fiber optic'))
fig.add_trace(go.Bar(
    x = [['Churn:No', 'Churn:No', 'Churn:Yes', 'Churn:Yes'],
        ['Female', 'Male', 'Female', 'Male']],
    y = [690, 717, 56, 57],
    name = 'No Internet'))

fig.update_layout(title_text= "<b>Churn Distribution w.r.t Internet Service and Gender</b>")
fig.data[0].marker.color = ('#7fcdff', '#7fcdff', '#7fcdff', '#7fcdff')
fig.data[1].marker.color = ('#326ada', '#326ada', '#326ada', '#326ada')
fig.data[2].marker.color = ('#ff9b35','#ff9b35','#ff9b35','#ff9b35')
fig.show()

Fiber optic service which was choosen by a lot of customers and it's evident that there's high churn rate among these customers. This could expose an issue in the Fiber optic service which dissatisfied most of it's customers, further looking into the issue might find a better and apt solution.
Customers who opted for DSL service are larger in number and is found to have less churn rate compared to Fibre optic service
# In[31]:


fig = px.histogram(df, x = 'Churn', color = 'Dependents', barmode = 'group', title = '<b>Churn distribution w.r.t. Dependents</b>')
fig.update_layout(width = 700, height = 500, bargap = 0.1)
fig.show()

Customers without dependents are more likely to churn
# In[32]:


color_map = {"Yes": "#7fcdff", "No": "#326ada"}
fig = px.histogram(df, x = 'Churn', color = 'Partner', barmode = "group", title = '<b>Churn distribution w.r.t. Partners</b>', color_discrete_map = color_map)
fig.update_layout(width = 700, height = 500, bargap = 0.1)
fig.show()


# In[33]:


color_map = {"Yes": '#7fcdff', "No": '#326ada'}
fig = px.histogram(df, x = 'Churn', color = 'SeniorCitizen', title = "<b>Churn distribution w.r.t Senior Citizen</b>", color_discrete_map = color_map)
fig.update_layout(width = 700, height = 500, bargap = 0.1)
fig.show()

On the above visual, a conclusion can be obtained such that customers without dependents and customers who have partners are more likely to churn while senior citizens being the most of churn.
# In[35]:


color_map = {"Yes": "#7fcdff", "No": "#326ada", "No internet service": "#ff9b35"}
fig = px.histogram(df, x = "Churn", color = "OnlineSecurity", barmode = "group", title = "<b>Churn w.r.t online security</b>", color_discrete_map = color_map)
fig.update_layout(width = 700, height = 500, bargap = 0.1)
fig.show()

Absence of online security,makes most customers churn.
# In[38]:


color_map = {"Yes": "#7fcdff", "No": "#326ada"}
fig = px.histogram(df, x = "Churn", color = "PaperlessBilling", barmode = "group", title = "<b>Churn distribution w.r.t Paperless Billing</b>", color_discrete_map = color_map)
fig.update_layout(width = 700, height = 500, bargap = 0.1)
fig.show()

Paperless Billing seems like one the reasons because of which customers are most likely to churn.
# In[39]:


color_map = {"Yes": "#7fcdff", "No": "#326ada", "No internet service": "#ff9b35"}

fig = px.histogram(df, x = 'Churn', color = 'TechSupport', barmode = 'group', title = '<b>Churn distribution w.r.t Techsupport</b>', color_discrete_map = color_map)
fig.update_layout(width = 700, height = 500, bargap = 0.1)
fig.show()

The absence of online security, Paperless Billing system and services with no TechSupport were the similiar trend are of the customers who are most likely churn.
# In[40]:


color_map = {"Yes": '#7fcdff', "No": '#326ada'}
fig = px.histogram(df, x = "Churn", color = "PhoneService", title = "<b>Churn distribution w.r.t Phone service</b>", color_discrete_map = color_map)
fig.update_layout(width = 700, height = 500, bargap = 0.1)
fig.show()

Eventhough there's a small fraction of customers but it's better to point out as they are more likely to churn because don't have a phone service.Conclusions as a Data Analyst :
● 75% of customer who have Month-to-Month Contract have opted to move out as compared to 13% of customers who have signed One Year Contract and 3% of customers who have signed Two Year Contract.
● Majority of the customers who moved out were having Electronic Check as Payment Method and others who opted for Credit-Card automatic transfer / Bank Automatic Transfer and Mailed Check as Payment Method were less likely to switch.
● Fiber optic service which was choosen by a lot of customers and it's evident that there's high churn rate among these customers. This could expose an issue in the Fiber optic service which dissatisfied most of it's customers, further looking into the issue might find a better and apt solution.
● Customers who opted for DSL service are larger in number and is found to have less churn rate compared to Fibre optic service
● Customers without dependents and customers who have partners are more likely to churn while senior citizens being the most of churn.
● The absence of online security, Paperless Billing system and services with no TechSupport were the similiar trend are of the customers who are most likely churn.
● There's a small fraction of customers who are more likely to churn and it's been found that they don't have a phone service.

A Data Analyst's work done here. Now,comes Data Scientist who makes a model to predict the churn in the future data.

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





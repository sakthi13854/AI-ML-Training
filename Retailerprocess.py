#!/usr/bin/env python
# coding: utf-8

# In[2]:


import logging
import glob


# In[37]:


logging.basicConfig(
    filename='ETLLOG.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
import os    
import pandas as pd
import numpy as np
import mysql.connector
import datetime


# In[20]:


csvs=glob.glob("C:/Users/Sakthi/Downloads/data sales/*.csv")
df=[pd.read_csv(file) for file in csvs]
df


# In[21]:


Df=pd.concat(df,ignore_index=True)
Df


# In[22]:


Df.dropna()


# In[23]:


Df['total_sales']=Df['Unit_Price']*Df['Quantity_Sold']


# In[24]:


Df


# In[51]:


Df['Date']=Df['Date'].dt.strftime('%y-%m-%d ')


# In[30]:


Df=Df.drop_duplicates(subset=['Store_ID','Date','Product_ID'])


# In[33]:


conditions=[
    Df['total_sales']>=1000,
    Df['total_sales']<1000,
]
choices=['High','low']
Df['Sales_category']=np.select(conditions,choices,default='low')


# In[52]:


Df


# In[44]:


# connecting with mydatabase
mydb=mysql.connector.connect(
    host='localhost',
    user='root',
    password='9625015',
    database='Retail_Sales'
)
cursor=mydb.cursor()


# In[47]:


cursor.execute("""CREATE TABLE retail_sales(
Store_ID VARCHAR(5),
Date date,
Product_ID VARCHAR(10),
Quantity_Sold int,
Discount_Percent int,
Payment_Mode VARCHAR(30),
total_sales float(2),
Sales_category VARCHAR(5))
""")


# In[59]:


cursor.execute("""ALTER TABLE retail_sales
ADD COlUMN Product_Name Varchar(30),
ADD COLUMN Unit_Price float(2);
""")


# In[61]:


sql = """
INSERT INTO retail_sales (
    Store_ID,
    Date,
    Product_ID,
    Product_Name,
    Quantity_Sold,
    Discount_Percent,
    Unit_Price,
    Payment_Mode,
    total_sales,
    Sales_category
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s,%s,%s)
"""

data=[tuple(row) for row in Df.values]
cursor.executemany(sql,data)


# In[62]:


mydb.commit()
cursor.close()
mydb.close()
logging.error("Something went wrong", exc_info=True)
print("ETL process completed successfully.")


# In[72]:


#using pandas to calculate Total sales per store.
# Top 5 products with the highest total sales.
# Daily total sales trend for each store.
TotalSales_per_Store=Df.groupby('Store_ID')['total_sales'].sum().reset_index()
High_total_values=Df.groupby('Product_ID')['total_sales'].sum().reset_index().sort_values(by='total_sales',ascending=False)
High5Products=High_total_values.head(5)
daily_trend=Df.groupby(['Store_ID','Date'])['total_sales'].sum().reset_index().sort_values(['Store_ID', 'Date'])
TotalSales_per_Store.to_csv('total sales per store.csv')
High5Products.to_csv('highest total score.csv')
daily_trend.to_csv('dailytrend.csv')


# In[75]:


import os
print(os.getcwd())
get_ipython().system('jupyter nbconvert --to script Retailerprocess.ipynb')


# In[ ]:





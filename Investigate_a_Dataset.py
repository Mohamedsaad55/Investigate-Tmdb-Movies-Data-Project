#!/usr/bin/env python
# coding: utf-8

# 
# 
# # Project: Investigate a Dataset - [TMDb Movies]
# 
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#Questions">Questions</a></li>    
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# <li><a href="#Data Limitation">Data Limitation</a></li>    
# </ul>

# <a id='intro'></a>
# ## Introduction
# 
# ### Dataset Description 
# 
# > **Tip**: In this section of the report, provide a brief introduction to the dataset you've selected/downloaded for analysis. Read through the description available on the homepage-links present [here](https://docs.google.com/document/d/e/2PACX-1vTlVmknRRnfy_4eTrjw5hYGaiQim5ctr9naaRd4V9du2B5bxpd8FEH3KtDgp8qVekw7Cj1GLk1IXdZi/pub?embedded=True). List all column names in each table, and their significance. In case of multiple tables, describe the relationship between tables. 
# 

# In[1]:


# Use this cell to set up import statements for all of the packages that you
#   plan to use.
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import csv 
import datetime as datetime


# <a id='wrangling'></a>
# ## Data Wrangling
# 

# In[2]:


# Load your data and print out a few lines. Perform operations to inspect data
#   types and look for instances of missing or possibly errant data.
df=pd.read_csv('Database_TMDb_movie_data/tmdb-movies.csv')
df.shape


# In[3]:


df.head(2) #checking columns and first 2 raws


# In[4]:


df.info() #checking data types


# In[5]:


df.nunique() #checking unique values


# In[6]:


df.describe()


# <a id='wrangling'></a>
# # Questions 
# <ul>
# <li><a href="#1">Which 5 movies had the highest and lowest profit?</a></li>
# <li><a href="#2">What`s the Highest and Lowest (Budget , Revenue?</a></li>
# <li><a href="#3">Relationship between Revenue , Budget and Profit</a></li>
# <li><a href="#4">Which movie had the greatest and least runtime?</a></li>
# <li><a href="#9">RelationShip between Profit and Vote Average</a></li>  
# <li><a href="#10">RelationShip between Vote Count and Vote Average</a></li>    
# <li><a href="#5">Relationship between RunTime and Vote Count </a></li>
# <li><a href="#6">Relationship between RunTime and Profit</a></li>
# <li><a href="#7">Which genres are most popular ?</a></li>
# <li><a href="#8">Which`re Top 5 Production Studios?</a></li>    
# </ul>
#   

# #  Data Cleaning
# 
#  

# #### let\`s check the Duplicates

# In[7]:


df.duplicated().sum() #checking Duplicates 


# In[8]:


df.drop_duplicates(inplace=True) #Droping Duplicates


# In[9]:


df.duplicated().sum() #checking duplicates after drop 


# #### Let\`s check the Columns and Drop the unused 

# In[10]:


df.columns #checking Columns names


# In[11]:


#Removing unwanted Columns 
df=df.drop(['imdb_id','homepage','tagline','overview','budget_adj','revenue_adj','keywords'],axis=1) 

print("Afetr Removing Unused Columns (Rows,Columns) : ",df.shape)


# In[12]:


df.info() #checking columns Types 


# In[13]:


df['release_date']=pd.to_datetime(df['release_date']) #Converting realease date Column type to Date time


# In[14]:


df['net_profit']=df['revenue']-df['budget'] #Creating New column (profit)


# In[15]:


df.head(2)


# #### Let\`s check the Na Values

# In[16]:


df.isnull().sum()


# In[17]:


# Columns we need to check for na
columns = ['budget', 'revenue']
# Replace 0 with NAN
df[columns] = df[columns].replace(0, np.NaN)
# Drop rows which contains NAN
df.dropna(subset = columns, inplace = True)
print("Afetr Droping rows contains NAN: ",df.shape)


# In[18]:


df.describe() #checking data


# <a id='eda'></a>
# ## Exploratory Data Analysis
# 
# > **Tip**: Now that you've trimmed and cleaned your data, you're ready to move on to exploration. **Compute statistics** and **create visualizations** with the goal of addressing the research questions that you posed in the Introduction section. You should compute the relevant statistics throughout the analysis when an inference is made about the data. Note that at least two or more kinds of plots should be created as part of the exploration, and you must  compare and show trends in the varied visualizations. 
# 
# 
# 
# > **Tip**: - Investigate the stated question(s) from multiple angles. It is recommended that you be systematic with your approach. Look at one variable at a time, and then follow it up by looking at relationships between variables. You should explore at least three variables in relation to the primary question. This can be an exploratory relationship between three variables of interest, or looking at how two independent variables relate to a single dependent variable of interest. Lastly, you  should perform both single-variable (1d) and multiple-variable (2d) explorations.
# 
# <a id='1'></a>
# ### Research Question 1 (Which 5 movies had the highest and lowest profit??)

# In[19]:


#top 5 movies
top_5=df.nlargest(5,'net_profit')  
top_5


# >As We can see Top 5 movies Made a Profit is : 
# 1. Avatar (With 2.544 Billion Dollars)
# 2. Star Wars: The Force Awakens (With 1.868 Billion Dollar)
# 3. Titanic (With 1.645 Billion Dollar)
# 4. Jurassic World (With 1.363 Billion Dollar)
# 5. Furious 7 (With 1.316 Billion Dollar)

# In[20]:


lowest_5=df.nsmallest(5,'net_profit') #lowest 5 movies
lowest_5


# >Now with the Lowest 5 Movies : 
# 1. The Warrior's Way (with -413 Million Dollars )
# 2. The Lone Ranger (with -165 Million Dollars)
# 3. The Alamo (With -119 Million Dollars)
# 4. Mars Needs Moms (With -111 Million Dollars)
# 5. Brother Bear (with -99 Million Dollar)

# <a id='2'></a>
# ### Research Question 2 (What\`s the Highest and Lowest (Budget , Revenue?) 

# In[21]:


def minmax(x):
    # function 'idmin' to find the lowest profit movie.
    min_index = df[x].idxmin()
    # function 'idmax' to find Highest profit movie.
    high_index = df[x].idxmax()
    high = pd.DataFrame(df.loc[high_index,:])
    low = pd.DataFrame(df.loc[min_index,:])
     # print the movie with high and low Budget and Revenue 
    print("Movie With the Highest "+ x + " : ",df['original_title'][high_index])
    print("Movie With the Lowest "+ x + "  : ",df['original_title'][min_index])
   
    return pd.concat([high,low],axis = 1)

# minmax function.
minmax('budget')


# In[22]:


minmax('revenue')


# In[23]:


minmax('net_profit')


# <a id='3'></a>
# ### Research Question 3 (Relationship between Revenue , Budget and Profit )

# In[24]:


# x-label and y-label
plt.xlabel('Revenue in Dollars')
plt.ylabel('Profit in Dollars')
# title
plt.title('Relationship between Revenue and Profit')
plt.scatter(df['revenue'], df['net_profit'],alpha=0.3)
plt.show()
# x-label and y-label
plt.xlabel('Budget in Dollars')
plt.ylabel('Profit in Dollars')
# title
plt.title('Relationship between Budget and Profit')
plt.scatter(df['budget'], df['net_profit'],alpha=0.3)
plt.show()


# >We can see that there\`s a strong relationship between profit and revenue, higher the profit, higher the revenue.
# ALSO We can see that there no as such relationship between budget and profits, But yes there are very less flims which didnt make profit when the budget was greater then 20M Dollar.

# <a id='4'></a>
# ### Research Question 4 (Which movie had the greatest and least runtime?)

# In[25]:


# first we can check the distribution of Runtime of all Movies with Histogram
plt.title('Runtime of all movies')
plt.hist(df['runtime'], bins = 200);
plt.show()


# In[26]:


# Runtime Average
df['runtime'].mean()


# >As we can see the Average runtime for All movies around 110 Minute

# In[27]:


df.nlargest(1,'runtime')


# >Movie with greatest runtime : 
# Carlos with 338 minutes record

# In[28]:


df.nsmallest(1,'runtime')


# > Movie with least runtime : Kid's Story As a 15 Min Record

# <a id='9'></a>
# ### Relationship Between Profit and Vote Average

# In[29]:


# x-label and y-label
plt.xlabel('Profit')
plt.ylabel('Vote Average')
# title
plt.title('Relationship between Profit and Vote average')
plt.scatter(df['net_profit'], df['vote_average'])
plt.show()


# <a id='10'></a> 
# ### RelationShip between Vote count and Vote Average
# 

# In[30]:


# x-label and y-label
plt.xlabel('Vote Count')
plt.ylabel('Vote Average')
# title
plt.title('Relationship between Vote Count and Vote average')
plt.scatter(df['vote_count'], df['vote_average'])
plt.show()


# <a id='5'></a>
# ### Research Question 5 (Relationship between RunTime and Vote Count )

# In[31]:


# x-label and y-label
plt.xlabel('Runtime in Minutes')
plt.ylabel('Vote Average')
# title
plt.title('Relationship between runtime and Vote average')
plt.scatter(df['runtime'], df['vote_count'])
plt.show()


# >As we can see that most votes goes to the Runtime average for all the movies that\`s around 110 Minutes 

# <a id='6'></a>
# ### Research Question 6 (Relationship between RunTime and Profit )

# In[32]:


# x-label and y-label
plt.xlabel('Runtime in Minutes')
plt.ylabel('Profit in (BillionDollars)')
# title
plt.title('Relationship between runtime and profit')
plt.scatter(df['runtime'], df['net_profit'],alpha=0.5)
plt.show()


# >Most of the movies have runtime in range of 85 to 130 Minutes

# <a id='7'></a>
# ### Research Question 7 (Which genres are most popular ?)

# In[33]:


genres_count = pd.Series(df['genres'].str.cat(sep = '|').split('|')).value_counts(ascending = False)
genres_count


# >So the Top 10 Genres are Drama, Comedy, Action, Thriller, Adventure, Romance, Crime, Family, Scince Fiction, Fantasy
# Lets visualize this with a plot

# In[34]:


# we can review the answer by diagram graph
diagram = genres_count.plot.bar()
# x-label and y-label
diagram.set_xlabel('Genres')
diagram.set_ylabel('Movies')
# title
diagram.set(title = 'Top Genres')
plt.show()


# <a id='8'></a>
# ### Research Question 8 (Which\`re Top 5 Production Studios?)  

# In[35]:


prod_company_count = pd.Series(df['production_companies'].str.cat(sep = '|').split('|')).value_counts(ascending = False)
top_5_studios=prod_company_count.head()
top_5_studios


# >As we can see the Top 5 Studios is :
# Universal Pictures  With  329 Movie
# Warner Bros.        With  324 Movie 
# Paramount Pictures  With  270 Movie
# Twentieth Century Fox Film Corporation  With  201 Movie 
# Columbia Pictures  With 178 Movie 
# ##### and here\`s a Plot to visualize the result

# In[36]:


# we can review the answer by diagram graph
diagram = top_5_studios.plot.barh()
# x-label and y-label
diagram.set_xlabel('Num. OF Movies')
diagram.set_ylabel('Production Comapnies')
# title
diagram.set(title = 'Top 5 Studios')
plt.show()


# >As we can see the Top production Company is Universal pictures 

# <a id='conclusions'></a>
# ## Conclusions
# 
# #### After investigating the Tmdb Movies data set we Figured out that 
# 1. Top 5 Movies made a profit is :
#    1. Avatar (With 2.544 Billion Dollars)
#    2. Star Wars: The Force Awakens (With 1.868 Billion Dollar)
#    3. Titanic (With 1.645 Billion Dollar)
#    4. Jurassic World (With 1.363 Billion Dollar)
#    5. Furious 7 (With 1.316 Billion Dollar)
# 2. Lowest 5 Profitable Movies :
#    1. The Warrior's Way (with -413 Million Dollars )
#    2. The Lone Ranger (with -165 Million Dollars)
#    3. The Alamo (With -119 Million Dollars)
#    4. Mars Needs Moms (With -111 Million Dollars)
#    5. Brother Bear (with -99 Million Dollar)
# 3. Movie With the Highest budget :  The Warrior's Way
# 4. Movie With the Lowest budget  :  Lost & Found
# 5. Average runtime for All movies around 110 Minute
# 6. Top 5 Geners for all the time is : 
#     1. Drama
#     2. Comedy
#     3. Thriller
#     4. Action
#     5. Adventure
# 7. Top 5 Production Companies    
#    1. Universal Pictures                        
#    2. Warner Bros.                              
#    3. Paramount Pictures                        
#    4. Twentieth Century Fox Film Corporation    
#    5. Columbia Pictures            
# 
# 

# <a id='Data Limitation'></a>
# ## Data Limitation
# 
# The conclusion is not full proof that given the above requirement the movie will be a big hit but it can be.
# 
# Also, we also lost some of the data in the data cleaning steps where we dont know the revenue and budget of the movie, which has affected our analysis.
# 
# This conclusion is not error proof.

# In[37]:


from subprocess import call
call(['python', '-m', 'nbconvert', 'Investigate_a_Dataset.ipynb'])


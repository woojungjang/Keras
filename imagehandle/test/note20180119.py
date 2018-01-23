
# coding: utf-8

# In[10]:


a = 1000000


# In[11]:


a + 50


# In[9]:


b = a + 50
b


# In[16]:


# 주석을 달아요.

# b는 a의 + 50결과이다.


# # 주석을 달아요.
# 메모를 작성해됴

# In[17]:


# file> save to check point> revert to check point 그지점으로 복귀


# In[22]:



get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np


# In[24]:


x = np.arange(-20, 20, 0.1)
y = np.sin(x)
plt.plot(x,y)


# In[25]:


jupyter nbconvert --to script [YOUR_NOTEBOOK].ipynb


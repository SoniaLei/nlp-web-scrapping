import pandas as pd
import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup

users = []
contents = []
tags = []

#Target company symbol (e.g. AAPL for Apple Inc.)
target = 'SPY'

# #Stocktwits login details
# username = 'nlpproject'
# password = 'Voorburg20!'

#webdriver for selenium automation.
#Ensure correct webdrivers are installed and pathway is correct.
driver = webdriver.Firefox(executable_path='C:\Program Files\Mozilla Firefox\geckodriver.exe')


#get target company feed directly
driver.get('https://stocktwits.com/symbol/'+target)


#Accept cookies
print('going to sleep')
time.sleep(15)
print('awake')
element = driver.find_element_by_id('onetrust-accept-btn-handler')
element.send_keys(Keys.ENTER)
print('cookies eaten')

#infinite scroll may work for powerful PCs, but doesn't seem to be an option here
#maybe attempt long scroll, append, and run the script again later?
#page down
no_of_pagedowns = 10
while no_of_pagedowns:
    element.send_keys(Keys.PAGE_DOWN)
    time.sleep(0.5)
    no_of_pagedowns-=1
    print('scrolling... ',no_of_pagedowns, ' to go')

content = driver.page_source
soup = BeautifulSoup(content, features="html.parser")
print('got soup')

#Find all tweets
findAllInSoup = soup.find_all(attrs={'class':'st_VgdfbpJ st_31oNG-n st_3A22T1e st_vmBJz6-'})

for a in findAllInSoup:
    userDiv = a.find('a', attrs={'class':'st_x9n-9YN st_2LcBLI2 st_1vC-yaI st_1VMMH6S'})
    user = userDiv.find('span').get_text()
    print(user)
    users.append(user)

    content = a.find('div', attrs={'class':'st_3SL2gug'}).get_text()
    print(content)
    contents.append(content)

driver.quit()
print('driver quit')

Table = {
        'User': users,
        'Tweet Contents': contents
        }
print('table laid')
print(Table)
df = pd.DataFrame(Table)

#df = pd.DataFrame({'User':users,'Tweet Contents':contents})
print('df written')
df.to_csv('tweets.csv', index=False, encoding='utf-8')
print('tweets.csv written')


input('Press ENTER to exit')

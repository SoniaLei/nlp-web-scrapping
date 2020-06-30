import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup

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

# driver.get("https://stocktwits.com/")

#Accept cookies
element = driver.find_element_by_id('onetrust-accept-btn-handler')
element.send_keys(Keys.ENTER)

# #Select 'Log in' button
# element = driver.find_element_by_xpath('/html/body/div[2]/div/div/div[1]/nav/div[3]/div/div/div[1]/button')
# element.send_keys(Keys.ENTER)
#
# #Enter login details
# element = driver.find_element_by_xpath('/html/body/div[2]/div/div/div[4]/div[2]/div/form/div[1]/div[1]/label/input')
# element.send_keys(username)
# element = driver.find_element_by_xpath('/html/body/div[2]/div/div/div[4]/div[2]/div/form/div[1]/div[2]/label/input')
# element.send_keys(password, Keys.ENTER)
#
# #Enter search terms (Company Symbol, e.g. AAPL for Apple Inc.)
# element = driver.find_element_by_xpath('/html/body/div[2]/div/div/div[2]/nav/div[2]/div/div[1]/div/input')
# print(element.text)
# # element.send_keys(target, Keys.ENTER)

content = driver.page_source
soup = BeautifulSoup(content)

findAllInSoup = soup.find_all(attrs={'class':'st_3SL2gug'})
#print('soup')
input('Press ENTER to exit')

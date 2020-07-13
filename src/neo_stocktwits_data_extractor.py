import os
import pandas as pd
import re
import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException
from bs4 import BeautifulSoup


def main():

    try:

        if not os.path.isdir('../data/raw'):
            os.mkdir('../data/raw')

        #data = []
        messageChecklist = []

        #Target company symbol (e.g. AAPL for Apple Inc.)
        target = 'SPY'

        # #Stocktwits login details
        # username = 'nlpproject'
        # password = 'Voorburg20!'

        #webdriver for selenium automation.
        #Ensure correct webdrivers are installed and pathway is correct.
        options = webdriver.FirefoxOptions()
        options.headless = False
        driver = webdriver.Firefox(options=options, executable_path='C:/Program Files/Mozilla Firefox/geckodriver.exe')


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

        time_between_refreshes = 30
        refresh_attempts = 120*70
        while refresh_attempts:

            data = []

            pageContent = driver.page_source
            soup = BeautifulSoup(pageContent, features="html.parser")
            print('got soup')

            #Find all tweets
            findAllInSoup = soup.find_all(attrs={'class':'st_VgdfbpJ st_31oNG-n st_3A22T1e st_vmBJz6-'})

            for a in findAllInSoup:

                messageIdA = a.find('a', attrs={'class': 'st_28bQfzV st_1E79qOs st_3TuKxmZ st_3Y6ESwY st_GnnuqFp st_1VMMH6S'})
                messageId = messageIdA.attrs['href']
                if messageId in messageChecklist:
                    print('No new tweets')
                    break

                sentimentSpan = a.find('span', attrs={'class':'st_11GoBZI'})
                if sentimentSpan is None:
                    continue
                sentimentDiv = sentimentSpan.find('div', attrs={'class': 'lib_XwnOHoV lib_3UzYkI9 lib_lPsmyQd lib_2TK8fEo'})
                sentiment = sentimentDiv.get_text()

                content = a.find('div', attrs={'class':'st_3SL2gug'}).get_text()
                if len(re.findall(r'\w+', content)) < 4:
                    continue

                userDiv = a.find('a', attrs={'class':'st_x9n-9YN st_2LcBLI2 st_1vC-yaI st_1VMMH6S'})
                user = userDiv.find('span').get_text()

                dateScraped = time.strftime("%Y/%m/%d", time.localtime())
                timeScraped = time.strftime("%H:%M:%S", time.localtime())

                print(f"Sentiment: {sentiment}\nUser: {user}\nMessageId: {messageId}\nContent: {content}\nDate: {dateScraped}\nTime: {timeScraped}\n")

                data.append((user, messageId, sentiment, content, dateScraped, timeScraped))
                messageChecklist.append(messageId)

            df = pd.DataFrame(data, columns=['user', 'message_id', 'sentiment', 'content', 'date', 'time'])

            print('table laid')

            if data:
                if not os.path.isfile('../data/raw/neoscrapedtweets.csv'):
                    df.to_csv('../data/raw/neoscrapedtweets.csv', index=False, encoding='utf-8')
                else: # else it exists so append without writing the header
                    df.to_csv('../data/raw/neoscrapedtweets.csv', mode='a', header=False, index=False)
                print(f"{len(df)} tweets written to neoscrapedtweets.csv")

            print('Going to sleep.', refresh_attempts, 'refreshes left.\n', len(messageChecklist), 'tweets scraped so far this session.')
            refresh_attempts-=1
            time.sleep(time_between_refreshes)

            if refresh_attempts % 20 == 0:
                driver.refresh()
                print('refreshing...')
                time.sleep(5)
            else:
                try:
                    element = driver.find_element_by_css_selector('.st_2t6tMpX.st_2-AYUR9.st_1Q0z4ky.st_jSJApJj.st_3pLPKgx.st_jGV698i.st_PLa30pM.st_1Z-amNw.st_1jzr122.st_2HqScKh')
                    element.click()
                except NoSuchElementException:
                    print('can''t refresh yet.')
                finally:
                    print('loading new posts...')

    finally:
        driver.close()
        driver.quit()
        print('driver quit')

    input('Press ENTER to exit')

if __name__ == '__main__':
    main()

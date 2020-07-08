import pandas as pd
import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException
from bs4 import BeautifulSoup


def main():

    try:

        data = []
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

        # content = driver.page_source
        # soup = BeautifulSoup(content, features="html.parser")
        # print('got soup')
        #
        # #Find all tweets
        # findAllInSoup = soup.find_all(attrs={'class':'st_VgdfbpJ st_31oNG-n st_3A22T1e st_vmBJz6-'})

        refresh_attempts = 10
        while refresh_attempts:

            content = driver.page_source
            soup = BeautifulSoup(content, features="html.parser")
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

                userDiv = a.find('a', attrs={'class':'st_x9n-9YN st_2LcBLI2 st_1vC-yaI st_1VMMH6S'})
                user = userDiv.find('span').get_text()

                content = a.find('div', attrs={'class':'st_3SL2gug'}).get_text()

                dateScraped = time.strftime("%Y/%m/%d", time.localtime())
                timeScraped = time.strftime("%H:%M:%S", time.localtime())

                print(f"Sentiment: {sentiment}\nUser: {user}\nMessageId: {messageId}\nContent: {content}\nDate: {dateScraped}\nTime: {timeScraped}\n")

                data.append((user, messageId, sentiment, content, dateScraped, timeScraped))
                messageChecklist.append(messageId)

            refresh_attempts-=1
            print('Going to sleep.', refresh_attempts, 'refreshes left.')
            time.sleep(20)

            try:
                element = driver.find_element_by_xpath('/html/body/div[2]/div/div/div[3]/div[2]/div/div[1]/div[2]/div/div/div[2]/div[2]/div/div')
                element.click()
            except NoSuchElementException:
                print('can''t refresh yet.')
            finally:
                print('refreshing...')

    finally:
        driver.close()
        driver.quit()
        print('driver quit')


    df = pd.DataFrame(data, columns=['user', 'message_id', 'sentiment', 'content', 'date', 'time'])

    print('table laid')

    df.to_csv('tweets.csv', index=False, encoding='utf-8')
    print(f"{len(df)} tweets written to tweets.csv")

    input('Press ENTER to exit')


if __name__ == '__main__':
    main()

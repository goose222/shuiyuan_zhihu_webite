#!/usr/bin/env python
#-*-coding:utf-8-*-
#@File:zhihu.py
import os
import time
from selenium import webdriver
import datetime
import psycopg2
import json

conn = psycopg2.connect(database="source_info", user="postgres", password="1", host="127.0.0.1", port="5432")##根据自己的数据库修改
cursor = conn.cursor()

def geturl():
    command = 'select url from zhihu_hottest_index;'
    cursor.execute(command)  ###最后自己根据题目建立数据库即可
    url_tumple = cursor.fetchall()
    cursor.close()
    conn.close()
    print(url_tumple)
    url=[]
    for i in url_tumple:
        url.append(i[0])
    return url

def scroll():  # 滚动条 可以滚动到底部不能再滚动为止，再爬取
    browser.execute_script('window.scrollBy(0,3000)')
    time.sleep(0.5)

def scroll_buttom(driver):
    # 定义一个初始值
    temp_height = 0
    n=500
    while n>=0:
        # 循环将滚动条下拉
        driver.execute_script("window.scrollBy(0,2000)")
        # sleep一下让滚动条反应一下
        time.sleep(1)
        # 获取当前滚动条距离顶部的距离
        check_height = driver.execute_script(
            "return document.documentElement.scrollTop || window.pageYOffset || document.body.scrollTop;")
        # 如果两者相等说明到底了
        if check_height == temp_height:
            break
        temp_height = check_height
        print(check_height)
        n=n-1
    return



# 设置浏览器需要打开的url
url = geturl()[0:]
#url = ["https://www.zhihu.com/topic/19575211"]
num=0
data=[]
crawl_date = str(datetime.date.today())
for j in url:
    print(num)
    num = num + 1
    # 引入chromedriver.exe
    driver = "D:\chromedriver_win32\chromedriver.exe"
    os.environ["webdriver.chrome.driver"] = driver

    # 设置为开发者模式
    options = webdriver.ChromeOptions()
    #options.add_argument("service_args=['–ignore-ssl-errors=true', '–ssl-protocol=TLSv1']")  # Python2/3
    options.add_experimental_option('excludeSwitches', ['enable-automation'])
    browser = webdriver.Chrome(driver, options=options)
    browser.maximize_window()

    print(j)
    browser.get(j+'/answers/updated')
    time.sleep(1)

    button=browser.find_element_by_xpath('/html/body/div[4]/div/div/div/div[2]/button').click()
    #滚动到底
    scroll_buttom(browser)

    page = browser.find_elements_by_xpath('//*[@id="QuestionAnswers-answers"]/div/div/div/div[2]/div[21]/button')[
        -2].text
    page = int(page)
    print(page)

    browser.quit()

    browser.get(j + '/answers/updated')
    time.sleep(1)

    button = browser.find_element_by_xpath('/html/body/div[4]/div/div/div/div[2]/button').click()
    # 滚动到底
    for x in range(page):
        scroll_buttom(browser)

        div = browser.find_element_by_class_name('Question-main')
        div = div.find_element_by_class_name('Question-mainColumn')
        div = div.find_element_by_xpath('div/div/div/div/div')
        div = div.find_elements_by_xpath('div')[1]
        rows = div.find_elements_by_xpath('div')
        print(rows[0].get_attribute('class'))
        print(len(rows))

        for i in rows:
            try:
                div=i.find_element_by_xpath('div')
                div=div.find_elements_by_xpath('div')[1]

                date=div.find_elements_by_xpath('div')[1]
                date=date.find_element_by_xpath('div/a/span').text

                thumbs_up_num = div.find_elements_by_xpath('div')[2]
                thumbs_up_num = thumbs_up_num.find_element_by_xpath('span/button').text[3:]

                reply_num = div.find_elements_by_xpath('div')[2].find_element_by_xpath('button').text[:-4]

                content=div.find_element_by_xpath('div/span')
                text = content.text
                content=content.find_elements_by_xpath('p')
                for k in content:
                    text=text+k.text
                print(date,thumbs_up_num,reply_num,text)
                data.append({'url':j,'content': text, 'thumbs_up_num': thumbs_up_num,'date':date,'crawl_date':crawl_date})

            except:
                try:
                    time.sleep(1)
                    page = browser.find_elements_by_xpath('//*[@id="QuestionAnswers-answers"]/div/div/div/div[2]/div[21]/button')[-1].click()
                    print(x)
                    continue
                except:
                    data = {'zhihu1': data}
                    with open("content"+str(num)+".json", 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                    break

    browser.quit()

#!/usr/bin/env python
#-*-coding:utf-8-*-
#@File:shuiyuan.py
import os
import time
from selenium import webdriver
import requests
import json
from selenium.webdriver.support.ui import Select
import  pytesseract
from PIL import Image
import datetime
date = datetime.date.today()
def scroll():  # 滚动条 可以滚动到底部不能再滚动为止，再爬取
    browser.execute_script('window.scrollBy(0,3000)')
    time.sleep(0.5)

def scroll_buttom(driver):
    # 定义一个初始值
    temp_height = 0

    while True:
        # 循环将滚动条下拉
        driver.execute_script("window.scrollBy(0,3000)")
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
    return


def get_code(page):
    image = Image.open(page)
    code = pytesseract.image_to_string(image, lang='eng')
    return code

def get_snap(driver):  # 对目标网页进行截屏。这里截的是全屏
    driver.save_screenshot('full_snap.png')
    page_snap_obj=Image.open('full_snap.png')
    return page_snap_obj

def get_image(driver):  # 对验证码所在位置进行定位，然后截取验证码图片
    img = driver.find_element_by_id('captcha-img')
    time.sleep(1)
    location = img.location
    print(location)
    size = img.size
    print(size)
    left = 1370
    top = 422
    right = left + 150
    bottom = top + 60

    page_snap_obj = get_snap(driver)
    image_obj = page_snap_obj.crop((left, top, right, bottom))
    #image_obj.show()
    return image_obj  # 得到的就是验证码

def getdata(browser):
    data = []
    # 输入用户名、密码
    user = browser.find_element_by_id('user')
    user.clear()
    user.send_keys('linduaner')

    password = browser.find_element_by_id('pass')
    password.clear()
    password.send_keys('linduaner8200347')

    # 填写验证码

    img = get_image(browser)

    path = 'code.png'
    img.save(path, 'png')

    captcha_code = get_code(path)
    print(captcha_code)

    captcha = browser.find_element_by_id('captcha')
    captcha.clear()
    captcha.send_keys(captcha_code)
    time.sleep(1)
    # 点击登录
    password = browser.find_element_by_id('submit-button').click()

    time.sleep(1)
    # 滚动到底
    scroll_buttom(browser)
    # topic-list-item (category-20-category tag-) ember-view##括号内，为变化的 xpath
    div = browser.find_element_by_id('ember59')  # 整个表
    whole_table = div.find_element_by_id('ember61')  # 整个表
    rows = whole_table.find_element_by_xpath('tbody')
    rows = rows.find_elements_by_xpath('tr')
    for row in rows:
        print(row)
        try:
            class_name = row.get_attribute('class')
            print(class_name[0:16])
            if class_name[0:16] != 'topic-list-item ':
                continue
            td = row.find_elements_by_xpath('td')
            span = td[0].find_element_by_xpath('span')
            content = span.find_element_by_xpath('a')
            print(content.text)
            title = content.text
            print(content.get_attribute('href'))
            url = content.get_attribute('href')
            ###title以及链接
            a = td[2].find_element_by_xpath('a')
            recall = a.find_element_by_xpath('span')
            print(recall.text)
            ###回复数
            view_num = td[3].find_element_by_xpath('span')
            print(view_num.text)
            ####浏览量
            new_time = td[4].get_attribute('title')
            print(new_time)
            new_time = new_time.split()
            print(new_time)
            old_time = new_time[1] + new_time[2]
            latest_time = new_time[4] + new_time[5]
            print(old_time, '1', latest_time)
            ##时间
            data.append(
                {'title': title, 'url': url, 'recall_num': recall.text, 'view_num': view_num.text, 'old-time': old_time,
                 'latest_time': latest_time, 'type': '学在交大', 'date': str(date)})
        except:
            continue

    data = {'sjtu_learning': data}
    with open("data1.json", 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    browser.quit()
    return

# 引入chromedriver.exe
driver = "C:/Users/63093/AppData/Local/Programs/Python/Python38/chromedriver.exe"
os.environ["webdriver.chrome.driver"] = driver


# 设置为开发者模式
options = webdriver.ChromeOptions()
options.add_argument("service_args=['–ignore-ssl-errors=true', '–ssl-protocol=TLSv1']") # Python2/3
options.add_experimental_option('excludeSwitches', ['enable-automation'])
browser = webdriver.Chrome(driver,options=options)
browser.maximize_window()
# 设置浏览器需要打开的url
url = "https://shuiyuan.sjtu.edu.cn/c/36-category/36"
browser.get(url)

try:
    getdata(browser)
except :
    getdata(browser)




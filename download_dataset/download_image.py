# coding:utf-8

import json
import itertools
import urllib
import requests
import os
import re
import sys
import datetime

str_table = {
    '_z2C$q': ':',
    '_z&e3B': '.',
    'AzdH3F': '/'
}

char_table = {
    'w': 'a',
    'k': 'b',
    'v': 'c',
    '1': 'd',
    'j': 'e',
    'u': 'f',
    '2': 'g',
    'i': 'h',
    't': 'i',
    '3': 'j',
    'h': 'k',
    's': 'l',
    '4': 'm',
    'g': 'n',
    '5': 'o',
    'r': 'p',
    'q': 'q',
    '6': 'r',
    'f': 's',
    'p': 't',
    '7': 'u',
    'e': 'v',
    'o': 'w',
    '8': '1',
    'd': '2',
    'n': '3',
    '9': '4',
    'c': '5',
    'm': '6',
    '0': '7',
    'b': '8',
    'l': '9',
    'a': '0'
}

# str 的translate方法需要用单个字符的十进制unicode编码作为key
# value 中的数字会被当成十进制unicode编码转换成字符
# 也可以直接用字符串作为value
char_table = {ord(key): ord(value) for key, value in char_table.items()}
# 解码图片URL
def decode(url):
    # 先替换字符串
    for key, value in str_table.items():
        url = url.replace(key, value)
    # 再替换剩下的字符
    return url.translate(char_table)

# 生成网址列表
def buildUrls(word):
    word = urllib.parse.quote(word)
    url = r"http://image.baidu.com/search/acjson?tn=resultjson_com&ipn=rj&ct=201326592&fp=result&queryWord={word}&cl=2&lm=-1&ie=utf-8&oe=utf-8&st=-1&ic=0&word={word}&face=0&istype=2nc=1&pn={pn}&rn=60"
    urls = (url.format(word=word, pn=x) for x in itertools.count(start=0, step=60))
    return urls

# 解析JSON获取图片URL
re_url = re.compile(r'"objURL":"(.*?)"')

def resolveImgUrl(html):
    imgUrls = [decode(x) for x in re_url.findall(html)]
    return imgUrls

def downImg(imgUrl, dirpath, imgName):
    filename = os.path.join(dirpath, imgName)
    try:
        res = requests.get(imgUrl, timeout=25)
        if str(res.status_code)[0] == "4":
            print(str(res.status_code), ":" , imgUrl)
            return False
    except Exception as e:
        print(" This is Exception：", imgUrl)
        print(e)
        return False

    with open(filename, "wb") as f:
        f.write(res.content)
    return True


def mkDir(dirName):
    download = os.path.join(sys.path[0], 'download')
    os.mkdir(download) if not os.path.exists(download)  else download
    dirpath = os.path.join(download, dirName)
    os.mkdir(dirpath) if not os.path.exists(dirpath) else dirpath #判断是否存在文件夹
    return dirpath

def crawl_data(urls, num):
    index = 0
    time_ms = datetime.datetime.now().strftime('%Y%m%d_%H%M_%S%f')  # 含微秒的日期时间
    for url in urls:
        print("requesting：", url)
        from urllib import request
        res = request.urlopen(url, timeout=30)  # 获取相应
        html = res.read()
        html = html.decode("utf-8")
        # print(html)
        # html = requests.get(url, timeout=10, allow_redirects=False).content.decode('utf-8') # 用上面urllib替换
        imgUrls = resolveImgUrl(html)
        if len(imgUrls) == 0:  # 没有图片则结束
            break
        for url in imgUrls:
            if downImg(url, dirpath, str(index) + '_' + time_ms + ".jpg"):
                index += 1
                print("正在下载第 %s  张图片" % index)
                if index==num:#最大下载图片数
                    return
            else:
                del_error = os.path.join(dirpath, str(index) + '_' + time_ms + ".jpg")
                try:
                    os.remove(del_error)
                except:
                    pass

    return

if __name__ == '__main__':

    print("做数据集请用-百度图片下载吧")
    print("Download in results")
    print("=" * 50)
    #word = input("Please input your word:\n")
    # name=input('请输入关键字：')
    # num = int(input('请输入你要下载的图片数量：'))
    names = ['暹罗猫', '布偶猫', '苏格兰折耳猫', '英国短毛猫', '波斯猫', '俄罗斯蓝猫', '美国短毛猫', '异国短毛猫', '挪威森林猫',
             '孟买猫', '缅因猫', '埃及猫', '伯曼猫', '斯芬克斯猫', '缅甸猫', '阿比西尼亚猫', '新加坡猫', '索马里猫', '土耳其梵猫', '中国狸花猫',
             '美国短尾猫', '西伯利亚森林猫', '日本短尾猫', '巴厘猫', '土耳其安哥拉猫', '褴褛猫', '东奇尼猫', '马恩岛猫', '柯尼斯卷毛猫',
             '奥西猫', '沙特尔猫', '德文卷毛猫', '呵叻猫', '美国刚毛猫', '重点色短毛猫', '哈瓦那棕猫', '波米拉猫', '塞尔凯克卷毛猫',
             '拉邦猫', '东方猫', '美国卷毛猫', '欧洲缅甸猫']
    num = 600
    for name in names:
        key_words = []
        key_words.append(name)
        for i in range(len(key_words)):
            word=key_words[i]
            print(word)
            dirpath = mkDir(word)

            urls = buildUrls(word)
            crawl_data(urls, num)

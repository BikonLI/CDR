import requests

def postNews():
    news = input("输入事件：")
    res = requests.post("http://49.233.183.144:11451/postnews/", json={"news": news})
    if res.text == "posed":
        print("发送成功！")
        
    else:
        print(res.status_code)
        print("发送失败")

def getNews():
    res = requests.get("http://49.233.183.144:11451/progress/")
    if res.status_code == 200:
        print(res.text)
    else:
        print("获取失败")
        
if __name__ == "__main__":
    print("服务器目前的消息如下：")
    getNews()
    if input("是否向服务器发送消息？ y/n") == 'y':
        postNews()
        

import time

localtime = time.asctime( time.localtime(time.time()) )
localtime=localtime.replace(" ","_")
print ("本地时间为 :", localtime)
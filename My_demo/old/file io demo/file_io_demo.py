
# 打开一个文件
fo = open("text.txt", "r+")#r+是读写模式，比较常用
print("文件名: ", fo.name)
print("是否已关闭 : ", fo.closed)
print("访问模式 : ", fo.mode)



# fo.write( "www.runoob.com!\nVery good site!\n")
str = fo.readline()#包括换行符
print("读取的字符串是 : ", str)








fo.close()
print("是否已关闭 : ", fo.closed)
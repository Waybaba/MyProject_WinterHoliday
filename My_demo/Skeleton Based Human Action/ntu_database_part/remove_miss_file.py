import shutil
import os


# os.chdir("/Users/Waybaba/Desktop")
# print(os.getcwd())
# shutil.move("/Users/Waybaba/Desktop/2/123","/Users/Waybaba/Desktop/1/123")

#creat and record source folder and destination folder

source_folder = "/Users/Waybaba/PycharmProjects/nturgb+d_skeletons/"
destination_folder = "/Users/Waybaba/PycharmProjects/nturgb+d_skeletons/miss_backup/"
#read the miss_file save all the file name into a tuple
fo = open("/Users/Waybaba/Library/Mobile Documents/com~apple~CloudDocs/科研/Machine Learning/MyProject/Skeleton Based Human Action/ntu_database_part/miss_list.txt")
all_lines = fo.readlines()
for index in range(len(all_lines)):
    all_lines[index] = all_lines[index][0:-1]+".skeleton"
fo.close()
#for each name in the tuple, make the full dir and move
for each_filename in all_lines :
    source_fullname = os.path.join(source_folder,each_filename)
    destination_fullname = os.path.join(destination_folder,each_filename)
    shutil.move(source_fullname,destination_fullname)







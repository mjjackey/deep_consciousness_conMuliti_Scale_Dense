import os
import sys
import csv

def write_file(file_path,text):
    with open(file_path,'a+',encoding='utf-8') as f:
        f.write(text+'\r\n')
        # f.write(text)

def write_csv(file_path,text):
    with open(file_path, mode='a',encoding='utf8',newline ='') as f:
        f = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        f.writerow(text)
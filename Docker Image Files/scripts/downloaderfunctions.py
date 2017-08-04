# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 16:02:32 2017

@author: visha
"""
import requests
from bs4 import BeautifulSoup
import zipfile
import os


# Downloads from a single link that is passed into the function.
def downloadExtractRemove(link,extractpath,filename,authcookie):
    filepath=extractpath+"/"+filename
    downfile = requests.get(link,cookies=authcookie)
    filelist = []
    with open(filepath, "wb") as fil:
        fil.write(downfile.content)
    with open(filepath, "rb") as file:
        zip_ref = zipfile.ZipFile(file)
        zip_ref.extractall(extractpath)
        z = zip_ref.filelist
        for f in z:
            filelist.append(f.filename)
        zip_ref.close()
    os.remove(filepath)
    return filelist

# Function gets all sample links
def getDownloadLinksFrom(filetype,searchList,downloadurl,username,password,authcookie):
    params = {"username":username,"password":password,"action":"acceptTandC","accept":"Yes","acceptSubmit":"Continue"}
    page = requests.post(downloadurl, data=params,cookies=authcookie)
    soup = BeautifulSoup(page.content, 'html.parser')
    downloadlinks=soup.find_all('a')
    downlink=""
    downfilename=""
    downfilesDict={}
    for link in downloadlinks:
        for filename in searchList:
            if (str(filename) in str(link.get_text())) & (filetype in str(link.get_text())):
                print("download from = "+link.attrs.get('href'))
                downlink=str(link.attrs.get('href'))
                downfilename=link.get_text()
                downfilesDict[downfilename]=downlink
                
    return downfilesDict


#Return current and next quarter list
def downloadCurrentAndNext(inputquarters):
    retlist=[]
    for quarters in inputquarters:
        year=int(quarters[2:])
        quarter=int(quarters[1])
        nextdownloadquarter = -1
        nextdownloadyear = -1
        if (year>=1999) & (year<=2016) & (quarter>0) & (quarter<=4):
            if quarter==4:
                nextdownloadquarter=1
                nextdownloadyear=year+1
            elif (quarter>0) & (quarter<4):
                nextdownloadyear=year
                nextdownloadquarter=quarter+1
        if (nextdownloadquarter > -1) & (nextdownloadyear > -1):
            retlist.append("Q"+str(nextdownloadquarter)+str(nextdownloadyear))
    return retlist


#Function used to get the session
def getSession(authurl,username,password):
    params={"username":username,"password":password,"action":"acceptTandC"}
    r = requests.post(authurl, data=params)
    opcookie=""
    for hist in r.history:
        for cookie in hist.cookies:
            if cookie.name=='PHPSESSID':
                opcookie=cookie.value
                print(cookie.value)
    authcookie = {}
    if opcookie != '':
        authcookie['PHPSESSID']=str(opcookie)
        authcookie['auth']='success'
    else:
        authcookie['auth']='error'
    return authcookie

# Function to logout from the session
def logout(authurl,username,password,authcookie):
    params={"username":username,"password":password,"action":"acceptTandC"}
    requests.post(authurl, data=params,cookies=authcookie)

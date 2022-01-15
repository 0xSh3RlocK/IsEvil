#!/usr/bin/env python3
import argparse
import time
import pandas as pd
import numpy as np
import random
# Machine Learning Packages
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from colorama import init, Fore
start_time = time.time()
urls_data = pd.read_csv("urldata.csv")

init()
GREEN = Fore.GREEN
RESET = Fore.RESET
GRAY = Fore.LIGHTBLACK_EX
BLUE = Fore.BLUE
WHITE = Fore.WHITE
RED = Fore.RED
x = Fore.CYAN
#
# print(type(urls_data))
#
# print(urls_data.head())
#

def makeTokens(f):
    tkns_BySlash = str(f.encode('utf-8')).split('/')	# make tokens after splitting by slash
    total_Tokens = []
    for i in tkns_BySlash:
        tokens = str(i).split('-')	# make tokens after splitting by dash
        tkns_ByDot = []
        for j in range(0,len(tokens)):
            temp_Tokens = str(tokens[j]).split('.')	# make tokens after splitting by dot
            tkns_ByDot = tkns_ByDot + temp_Tokens
        total_Tokens = total_Tokens + tokens + tkns_ByDot
    total_Tokens = list(set(total_Tokens))	#remove redundant tokens
    if 'com' in total_Tokens:
        total_Tokens.remove('com')	#removing .com since it occurs a lot of times and it should not be included in our features
    return total_Tokens



# print("Accuracy ",logit.score(X_test, y_test))

#
# X_predict = ["google.com/search=jcharistech",
# "google.com/search=faizanahmad",
# "pakistanifacebookforever.com/getpassword.php/",
# "www.radsport-voggel.de/wp-admin/includes/log.exe",
# "ahrenhei.without-transfer.ru/nethost.exe ",
# "www.itidea.it/centroesteticosothys/img/_notes/gum.exe"]
# X_predict = ["www.itidea.it/centroesteticosothys/img/_notes/gum.exe"]


if __name__ == '__main__':
    try:
        y = urls_data["label"]
        url_list = urls_data["url"]
        vectorizer = TfidfVectorizer(tokenizer=makeTokens)
        X = vectorizer.fit_transform(url_list)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model Building
        # using logistic regression
        logit = LogisticRegression(solver='lbfgs', max_iter=1000)
        logit.fit(X_train, y_train)
        parser = argparse.ArgumentParser(prog='Anubis.py')
        parser.add_argument("--URL", "-u",help="Scan The url")
        args = parser.parse_args()
        if args.URL:
            X_predict = [f'{args.URL}']
            X_predict = vectorizer.transform(X_predict)
            New_predict = logit.predict(X_predict)
            if(New_predict[0] == 'bad'):
                print(f'\n{args.URL} is {RED}{New_predict[0]} SITE IT IS NOT SAFE TO ENTER \n')
            else:
                print(f'\n{args.URL} is {GREEN}{New_predict[0]} SITE IT IS SAFE TO ENTER \n')
        print(f"{WHITE}Accuracy ", logit.score(X_test, y_test))
        print(f'{WHITE}Finished in {round(time.time() - start_time, 3)} Seconds')
    except KeyboardInterrupt:
        print('You pressed CTRL + C Bye!!!!!')


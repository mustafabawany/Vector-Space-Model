from flask import Flask, render_template, url_for, request
from flask_sqlalchemy import SQLAlchemy
import nltk
import os
import math
import re
import collections 
from numpy import dot
from numpy.linalg import norm
from array import *
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
app = Flask(__name__)


@app.route("/" , methods=['POST' , 'GET'])

def test():
    if request.method == "POST":
        Query = request.form['query']
        QueryProcessed1 = Query.lower()

        """Removing Punctuations"""
        QueryProcessed2 = removePunctuations(QueryProcessed1)

        """Removing Stop Words"""
        QueryProcessed3 = removeStopWords(QueryProcessed2)
        
        """Applying Porter Stemmer"""
        QueryProcessed4 = stemSentence(QueryProcessed3)
        
        """Seperating Hyphenated words"""
        QueryProcessed5 = removeHyphenatedWords(QueryProcessed4)
        
        """Creating Inverted Index"""
        #Assuming 0th document is the query 
        createDictionary(QueryProcessed5, 0)
        
        """Creating Term Frequency Matrix"""
        #Assuming 0th document is the query 
        createTermFrequency(QueryProcessed5, 0)

        """----------------------Looking for common words in document and query------------------"""
        
        #Creating a Matrix to store frequencies of all the terms
        Tf = [[0 for i in range(450)] for j in range (len(Dictionary))]

        termNo = 0
        for term in Dictionary.keys():
            for doc in TermFrequency.keys():
                tempDict = TermFrequency[doc]
                if term in tempDict: 
                    frequency = tempDict[term]
                    Tf[termNo][doc] = frequency 
            termNo = termNo + 1

        #Now have to find the total term frequency in all the documents to calculate the idf
        #Calculating Total Term Frequency of Each Term

        #Calculating Tf
        for i in range(len(Dictionary)):
            freq = 0
            for j in range(449):
                if Tf[i][j] > 0:
                    freq = freq + 1
                
            Tf[i][449] = math.log(449/freq, 10)
            
        Tf_IDF = [[0 for i in range (len(Dictionary))] for j in range(449)]

        #Calculating IDF
        for i in range(449):
            for j in range(len(Dictionary)):
                Tf_IDF[i][j] = round(Tf[j][i] * Tf[j][449], 1)

        Result = [0 for i in range(449)]
        
        #Calculating Cosine Similarity
        i = 0
        for row in Tf_IDF:
            if i == 449:
                break;
            if i != 0:                                                  #Because document 0 is Query
                Result[i] = dot(Tf_IDF[0] , row) / (norm(Tf_IDF[0]) * norm(row))
            i = i + 1

        i = 0
        docs = []
        for row in Result:
            if (row > alpha):
                docs.append(i)
            i = i + 1

        resultDictionary = dict()
        
        for docNo in docs:
            with open(os.getcwd() + "/Abstracts/" + str(docNo) + ".txt", encoding="utf8" , errors="ignore") as fileHandle:
                plainSentence = fileHandle.read()
                resultDictionary[str(docNo)] = plainSentence
        return render_template('result.html' , result=resultDictionary , Query=Query)
    else :
        for docNo in range (1,449):
            docNo = str(docNo)
            with open(os.getcwd() + "/Abstracts/" + docNo + ".txt", encoding="utf8" , errors="ignore") as fileHandle:
                plainSentence = fileHandle.read()
            
            """----------------------------------Pre Processing Phase------------------------------"""
            
            """Lower casing words."""
            plainSentence = plainSentence.lower()

            """Removing punctuations."""
            preProcess1 = removePunctuations(plainSentence)

            """Removing Stop Words."""
            preProcess2 = removeStopWords(preProcess1)

            """Applying Porter Stemmer."""
            preProcess3 = stemSentence(preProcess1)

            """Seperating Hyphenated words."""
            preProcess4 = removeHyphenatedWords(preProcess3)

            """Creating Dictionary."""
            createDictionary(preProcess4, int(docNo))
           
            """Creating Term Frequency Matrix"""
            createTermFrequency(preProcess4, int(docNo))
        
        return render_template('index.html')

if __name__ == "__main__":
    ps = PorterStemmer()
    Dictionary = dict()
    TermFrequency = dict()
    alpha = 0.001

    def stemSentence(sentence):
        token_words = word_tokenize(sentence)
        stem_sentence=[]
        for word in token_words:
            stem_sentence.append(ps.stem(word))
            stem_sentence.append(" ")
        return "".join(stem_sentence)

    def removeStopWords(sentence):
        stopWordsList = []
        tempFileHandle = open(os.getcwd() + "/Stopword-List.txt" , "r");
        stopWords = tempFileHandle.read()
        token_stopWord = word_tokenize(stopWords)
        for eachStopWord in token_stopWord:
            stopWordsList.append(eachStopWord)
        token_words = word_tokenize(sentence)
        resultSentence  = [word for word in token_words if word not in stopWordsList]
        resultSentence = ' '.join(resultSentence)
        return resultSentence

    def removePunctuations(sentence):
        punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
        newSentence = ""
        for word in sentence:
            if (word in punctuations):
                newSentence = newSentence + " "
            else: 
                newSentence = newSentence + word

        return newSentence

    def removeHyphenatedWords(sentence):
        token_words = word_tokenize(sentence)
        newSentence = []
        for word in token_words:
            if '-' in word:
                temp = []
                for character in word:
                    if character == '-':
                        temp.append(" ")
                    else :
                        temp.append(character)
                temp = "".join(temp)
                newSentence.append(temp)
                newSentence.append(" ")
            else: 
                newSentence.append(word) 
                newSentence.append(" ")
        return "".join(newSentence)

    """------------------------------------------Term Frequency--------------------------------------"""

    def createTermFrequency(sentence , docNo):
        tempDict = dict()
        token_words = word_tokenize(sentence)
        for word in token_words:
            if word in tempDict:
                count = tempDict[word]
                tempDict[word] = count + 1
            else:
                tempDict[word] = 1
        TermFrequency[docNo] = dict(sorted(tempDict.items()))

    """------------------------------------------Inverted Index--------------------------------------"""
    def createDictionary(sentence , docNo):
        token_words = word_tokenize(sentence)
        for word in token_words:
            if word not in Dictionary:
                docList = []
                docList.append(docNo)
                Dictionary[word] = docList
            else:
                predocList = Dictionary[word]
                if docNo not in predocList:
                    docList = predocList
                    docList.append(docNo)
                    Dictionary[word] = docList

    def searchInDictionary(Query):
        total_list = []
        token_words = word_tokenize(Query)
        for word in token_words:
            if word in Dictionary:
                tempList = Dictionary[word]
                total_list.append(tempList)
            else:
                print("Word not in dictionary")
        return total_list
    app.run(debug=True)
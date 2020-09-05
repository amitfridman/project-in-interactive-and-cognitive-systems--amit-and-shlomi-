import nltk
import os
import random
import numpy as np
from gensim.models import Word2Vec #we import the word2vec model



def save_tokens_to_text_file(documents_tokens,file_path):

    output_file=open(file_path,'w')
    for token in documents_tokens:
        output_file.write(token)
        output_file.write(" \n")



#the following help function remove numbers,dates
#According the article we leave only:
#""Then, we filter the tokenized data by retaining
#only tokens composed of the following four types of characters: alphabetic, hyphen, dot
#and apostrophe, and containing at least one alphabetic character.""
#Moreover, we take at most 400 tokens
def Filter_text(list_of_tokens):
   filtered_tokens=[]#this tokens will conclude only alphabetic characters
   for word  in list_of_tokens:
       if any(c.isalpha() for c in word):#we remove from the list and we do not return the new one
            filtered_tokens.append(word)
       elif word=='.':
            filtered_tokens.append(word)
       if len(filtered_tokens)==400:#we take this number according our article.
            return filtered_tokens
    #In the case that the filtered doucment is shorter than 400 words we will add dumy character *
   if len(filtered_tokens)<400:
        while len(filtered_tokens)<400:
            filtered_tokens.append('*')
   return filtered_tokens

#In this file we preprocess the text data
if __name__ == "__main__":

    article_groups_directories=os.listdir('20_newsgroups')#In each directory
    pairs_tokens_labels=[]
    #each article has an appropriate label.
    #i is the index for the dcuments label
    j=0
    for i,dir in enumerate(article_groups_directories):
        post_names=os.listdir('20_newsgroups\\'+dir)
        #we loop over all documents in the specific directory
        for name in post_names:
            post=open('20_newsgroups\\'+dir+'\\'+name,'r',encoding='utf-8',errors='ignore')
            text=post.read()
            #now we remove the header
            without_header = text.split('\n\n', 1)[1]
            tokens= list(nltk.word_tokenize(without_header))
            filtered_tokens=Filter_text(tokens)#we got the documents tokens.
            #we add the filtered_tokens of the current document to the list
            pairs_tokens_labels.append((filtered_tokens,str(i)))#str(i) is the label of the document
            j=j+1
            print(j)
            #we transform it to the string form


   #we randomize the dataset
    random.shuffle(pairs_tokens_labels)
    #we split the dataset to train and test

    #recall: pairs_tokens_labels[i][0] the document tokens
    # pairs_tokens_labels[i][1] is the label string ,for example '2'.
    for i in range(0,15000):#saves representations to train set

        document_tokens=pairs_tokens_labels[i][0]#we take the list of tokens for the specif documents
        file_name='doc_'+str(i)+'_label_'+pairs_tokens_labels[i][1]+'.txt'#we create the path
        save_tokens_to_text_file(document_tokens, 'Train_tokens\\' + file_name)


    for i in range(5000,len(pairs_tokens_labels)):#saves representations to train set
        document_tokens = pairs_tokens_labels[i][0]  # we take the list of tokens for the specif documents
        file_name = 'doc_' + str(i) + '_label_' + pairs_tokens_labels[i][1] + '.txt'  # we create the file path
        save_tokens_to_text_file(document_tokens,'Test_tokens\\'+file_name)



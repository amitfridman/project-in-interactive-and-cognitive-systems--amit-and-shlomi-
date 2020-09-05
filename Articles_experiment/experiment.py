import os
from collections import defaultdict
from torchtext.vocab import Vocab
from torch.utils.data.dataset import Dataset, TensorDataset
from pathlib import Path
from collections import Counter
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import time
import os
from torch.utils.data.dataloader import DataLoader
from matplotlib import pyplot as plt
import nltk



# The only special token is unknown token for the embedding
UNKNOWN_TOKEN = "<unk>"
SPECIAL_TOKENS = [ UNKNOWN_TOKEN]



def evaluate(test_dataloader, model, test):
    floss = nn.NLLLoss(ignore_index=-1)
    start = time.time()
    acc = 0
    loss_avg = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, input_data in enumerate(test_dataloader):
            words_idx_tensor, labels = input_data
            outputs = model(words_idx_tensor)
            _,predicted=torch.max(outputs.data,1)
            #print(predicted,labels)
            loss_avg += floss(outputs,labels)
            if predicted==labels:
                acc += 1
        acc = acc / len(test)
        print('test_eval_time:', time.time() - start)
    return acc, loss_avg / len(test)

# Here we create a total corpus of all words in the training datasets
#the training document are given as a text file of 400 filtered tokens
def get_vocabs(dir_path):
    word_dict = defaultdict(int)  # dicts which values are dict
    documents_names=os.listdir(dir_path)
    for document in documents_names:
         doc=open(dir_path+'\\'+document,'r',encoding='utf-8',errors='ignore')
         text=doc.read()
         tokens = text.split('\n')[:-1]
         #print(tokens)
         if len(tokens)>400:
             print(tokens)
         for token in tokens:
             word_dict[token]+=1

    return word_dict

#question is the parameter is all directions all just the part of them
#how do we deal with the label?we do not need it for the embedding,
#
class Documents_reader:
    def __init__(self, dir_path, word_dict):
        self.dir_path = dir_path
        self.word_dict = word_dict

        self.documents = []
        self.labels=[]
        self.__readData__()

    def __readData__(self):
        documents_names = os.listdir(self.dir_path)
        for document in documents_names:
            doc = open(self.dir_path + '\\' + document, 'r', encoding='utf-8', errors='ignore')
            # now we remove the header
            text = doc.read()
            tokens = text.split('\n')[:-1]
            self.labels.append(int(document.split("_")[3][:-4]))
            self.documents.append(tokens)#we apped the tokens of the current documents
        print(self.labels)


class Dataset_model(Dataset):
    def __init__(self, word_dict,  subset: str,  word_embeddings=None):
        super().__init__()
        self.subset = subset  # One of the following: [Train, test]
        if subset == 'Train':
            self.dir = subset + "_tokens"
            self.datareader = Documents_reader(self.dir, word_dict)
        else:
            self.dir = subset + "_tokens"
            self.datareader = Documents_reader(self.dir, word_dict)
        self.vocab_size = len(self.datareader.word_dict)
        if word_embeddings:
            self.word_idx_mappings, self.idx_word_mappings, self.word_vectors = word_embeddings
        else:
            self.word_idx_mappings, self.idx_word_mappings, self.word_vectors = self.init_word_embeddings(
                self.datareader.word_dict)
        self.unknown_idx = self.word_idx_mappings.get(UNKNOWN_TOKEN)
        self.word_vector_dim = self.word_vectors.size(-1)
        self.documents_dataset = self.convert_documents_to_dataset()

    def __len__(self):
        return len(self.documents_dataset)

    def __getitem__(self, index):
        word_embed_idx, labels = self.documents_dataset[index]
        return word_embed_idx,  labels

    @staticmethod
    def init_word_embeddings(word_dict):
        glove = Vocab(Counter(word_dict), vectors="glove.6B.300d", specials=SPECIAL_TOKENS)
        return glove.stoi, glove.itos, glove.vectors

    def get_word_embeddings(self):
        return self.word_idx_mappings, self.idx_word_mappings, self.word_vectors


    def convert_documents_to_dataset(self):
        document_word_idx_list = list()
        document_labels = self.datareader.labels

        for document_idx, document in enumerate(self.datareader.documents):
            words_idx_list = []
            # label_list = []
            for word in document:
                if word not in self.datareader.word_dict:
                    words_idx_list.append(self.word_idx_mappings.get('<unk>'))
                else:
                    words_idx_list.append(self.word_idx_mappings.get(word))
                # label_list.append(int(label))

            # if padding:
            #     while len(words_idx_list) < self.max_seq_len:
            #         words_idx_list.append(self.word_idx_mappings.get(PAD_TOKEN))
            #         pos_idx_list.append(self.pos_idx_mappings.get(PAD_TOKEN))
            document_word_idx_list.append(torch.tensor(words_idx_list, dtype=torch.long, requires_grad=False))

            # sentence_labels.append(torch.tensor(label_list, dtype=torch.long, requires_grad=False))


        # if padding:
        #     all_sentence_word_idx = torch.tensor(sentence_word_idx_list, dtype=torch.long)
        #     all_sentence_pos_idx = torch.tensor(sentence_pos_idx_list, dtype=torch.long)
        #     all_sentence_len = torch.tensor(sentence_len_list, dtype=torch.long, requires_grad=False)
        #     return TensorDataset(all_sentence_word_idx, all_sentence_pos_idx, all_sentence_len)

        return {i: sample_tuple for i, sample_tuple in enumerate(zip(document_word_idx_list,document_labels))}

# we have defined the loss function manually
def Loss(edges_score, true_labels):
    l = torch.zeros(len(true_labels))  # l is the loss score
    # score=nn.LogSoftmax(edges_score,dim=1)# edge score is the an array of  of exp of the edge values
    for i in range(1, len(edges_score)):  # in this loop we calculate the probabilities and add it to the sum
        l[i] = -(1 / len(true_labels[1:])) * edges_score[true_labels[i]][i]
    # print('l',l)
    return torch.sum(l)


class articlenet(nn.Module):
    def __init__(self, word_embedding_dim, word_vocab_size):
        super(articlenet, self).__init__()
        self.word_embedding = nn.Embedding(word_vocab_size, word_embedding_dim)
        self.hidden_dim = self.word_embedding.embedding_dim
        self.conv=nn.Conv1d(400,30,kernel_size=3)
        self.relu=nn.ReLU()
        self.linear=nn.Linear(8940,20)
        self.dropout = nn.Dropout()
        self.log_softmax = nn.LogSoftmax()
        # self.loss_function = nn.NLLLoss(ignore_index=-1)# Implement the loss function described above

    def forward(self, sentence):
        word_idx_tensor = sentence
        # Pass word_idx and pos_idx through their embedding layers
        x_embeds = self.word_embedding(word_idx_tensor)
        # Concat both embedding outputs
        #print(x_embeds.shape)
        v = self.conv(x_embeds)
        #print(v.shape)
        v=v.view((v.shape[0],-1))
        v=self.dropout(v)
        v=self.relu(v)
        v=self.linear(v)
        #print(v.shape)
        v=self.log_softmax(v)
        return v






if __name__ == '__main__':

    Train_path = "Train_tokens"
    print("path_train -", Train_path)
    path_test = "Test_tokens"
    print("path_test -", path_test)

    paths_list = Train_path
    word_dict = get_vocabs(Train_path)#We create the vocabulary according the Trainset
    train = Dataset_model(word_dict,  subset='Train')
    train_dataloader = DataLoader(train,batch_size=100, shuffle=True)
    test = Dataset_model(word_dict,  subset='Test')
    test_dataloader = DataLoader(test, shuffle=False)
    print(len(train_dataloader), len(test_dataloader))

    EPOCHS = 20
    WORD_EMBEDDING_DIM = 300
    POS_EMBEDDING = 25
    HIDDEN_DIM = 125
    word_vocab_size = len(train.word_idx_mappings)


    # The first model is the advance and the other is the regular one.
    model = articlenet( WORD_EMBEDDING_DIM, word_vocab_size)

    # model = KiperwasserDependencyParser(POS_EMBEDDING, WORD_EMBEDDING_DIM, word_vocab_size, tag_vocab_size, HIDDEN_DIM)

    # Define the loss function as the Negative Log Likelihood loss (NLLLoss)
    # loss_function = nn.NLLLoss()

    # We will be using a simple SGD optimizer to minimize the loss function
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    acumulate_grad_steps = 50  # This is the actual batch_size, while we officially use batch_size=1

    # Training start
    print("Training Started")
    floss = nn.NLLLoss(ignore_index=-1)
    accuracy_list = []
    loss_list = []
    test_loss = []
    acc_test = []
    epochs = EPOCHS
    start = time.time()
    max_acc = 0
    for epoch in range(epochs):
        acc = 0  # to keep track of accuracy
        loss1 = 0  # To keep track of the loss value
        i = 0
        model.train()
        for batch_idx, input_data in enumerate(train_dataloader):
            i += 1
            words_idx_tensor,labels = input_data

            outputs = model(words_idx_tensor)
            #print(outputs.shape,labels.shape)
            _, predicted = torch.max(outputs.data, 1)
            loss = floss(outputs,labels)
            """
            tag_scores = tag_scores.unsqueeze(0).permute(0, 2, 1)
            # print("tag_scores shape -", tag_scores.shape)
            # print("pos_idx_tensor shape -", pos_idx_tensor.shape)
            """
            loss.backward()
            optimizer.step()
            model.zero_grad()
            acc += (predicted==labels).sum().item()
            loss1+=loss

        loss = loss1/len(train)
        acc = acc / len(train)
        loss_list.append(float(loss))
        accuracy_list.append(float(acc))
        test_acc, loss_test = evaluate(test_dataloader, model, test)
        test_loss.append(loss_test)
        acc_test.append(test_acc)
        e_interval = i
        print("Epoch {} Completed,\tLoss {}\tAccuracy: {}\t Test Accuracy: {}".format(epoch + 1,
                                                                                      np.mean(loss_list[-e_interval:]),
                                                                                      np.mean(
                                                                                          accuracy_list[-e_interval:]),
                                                                                      test_acc))
        if test_acc > max_acc:
            torch.save(model.state_dict(), 'model_advanced.pkl')
            max_acc = test_acc

    plt.plot(np.arange(1, epochs + 1), test_loss, label='test')
    plt.plot(range(1, epochs + 1), loss_list, label='train')
    plt.title('Loss as Function of Number of Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.plot(np.arange(1, epochs + 1), acc_test, label='test')
    plt.plot(range(1, epochs + 1), accuracy_list, label='train')
    plt.title('Accuracy as Function of Number of Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('UAS')
    plt.legend()
    plt.show()


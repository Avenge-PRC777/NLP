'''
Author: Prasanjit Rath
Date of creation: March 31, 2022
'''

import os
import torch
import string
import unicodedata
import torch.nn as nn
import math
import random
import matplotlib.pyplot as plt
random.seed(42)

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        '''
        nn.Linear(m,n) initializes a weight matrix of size m X n that is multiplied with
        input matrix to give output. So input should be a X m, then output is a X n
        '''
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim = 1)
    
    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden
    
    def initHidden(self):
        # Very first hidden state which is 0
        return torch.zeros(1, self.hidden_size)

class RNNClassifier:
    def __init__(self, n_hidden = 128, learning_rate = 0.001, train_test_ratio = 0.75):
        self.cwd = os.getcwd()
        self.data_dir = os.path.join(self.cwd, "data")
        self.names_dir = os.path.join(self.data_dir, "names")
        self.data_dictionary = {}
        self.ascii_letters = string.ascii_letters + " .,;'"
        self.letter_tensor_len = len(self.ascii_letters)
        self.n_hidden = n_hidden
        self.training_list = []
        self.testing_list = []
        self.train_test_ratio = train_test_ratio
        self.criterion = nn.NLLLoss()
        self.learning_rate = learning_rate
        self.loss_values = []
    
    def set_rnn_module(self):
        self.rnn_module = RNN(self.letter_tensor_len, self.n_hidden, self.get_num_of_categories())
    
    def get_rnn_module(self):
        return self.rnn_module
    
    def get_letter_tensor_len(self):
        return self.letter_tensor_len
    
    def get_num_of_categories(self):
        return len(self.data_dictionary.keys())

    def get_training_list(self, top_k = -1):
        if top_k < 1:
            return self.training_list
        else:
            p = {}
            count = 0
            for key in self.training_list.keys():
                p[key] = self.training_list[key]
                count += 1
                if count == top_k:
                    return p

    def set_data_dictionary(self):
        file_list = os.listdir(self.names_dir)
        file_list.sort()
        self.sorted_categories = [e[:-4] for e in file_list]
        # We take only 2 categories, making it binary classification
        self.sorted_categories = self.sorted_categories[0:2]
        for itern, txt_file in enumerate(self.sorted_categories):
            surnames = open(os.path.join(self.names_dir, txt_file + ".txt"), encoding = 'utf-8').read().strip().split('\n')
            surnames = [self.unicode2ascii(e) for e in surnames]
            for surname in surnames[0:math.floor(len(surnames)*self.train_test_ratio)]:
                self.training_list.append((self.word2vec(surname), torch.tensor([itern])))
            for surname in surnames[math.floor(len(surnames)*self.train_test_ratio):]:
                self.testing_list.append((self.word2vec(surname), torch.tensor([itern])))
            if txt_file not in self.data_dictionary:
                self.data_dictionary[txt_file] = surnames
    
    def get_data_dictionary(self):
        return self.data_dictionary
    
    def unicode2ascii(self, word):
        '''
        This function is required to normalize letters with diacritics into ascii so that
        we can vectorize every letter as a vector of 52 (26-uppercase, 26-lowercase)
        '''
        return ''.join(letter for letter in unicodedata.normalize('NFD', word) if unicodedata.category(letter) != 'Mn' and letter in self.ascii_letters)
    
    def word2vec(self, word):
        word_tensor = torch.zeros(len(word), 1, self.letter_tensor_len)
        for itern, letter in enumerate(word):
            word_tensor[itern][0][self.ascii_letters.find(letter)] = 1
        return word_tensor
    
    def train_one_epoch(self):
        hidden = self.rnn_module.initHidden()
        self.rnn_module.zero_grad() # Sets gradients of all model parameters to 0
        random.shuffle(self.training_list)
        self.training_list = self.training_list[0:100]
        total_loss = 0
        for train_data, category in self.training_list:
            for letter_vector_num in range(train_data.size()[0]):
                output, hidden = self.rnn_module(train_data[letter_vector_num], hidden)
            loss = self.criterion(output, category)
            loss.backward(retain_graph = True)
            
            for parameter in self.rnn_module.parameters():
                parameter.data.add_(parameter.grad.data, alpha = -self.learning_rate)
            
            total_loss += loss.item()
        
        recall = 0
        hidden2 = self.rnn_module.initHidden()
        random.shuffle(self.testing_list)
        self.testing_list = self.testing_list[0:10]
        for test_data, category_idc in self.testing_list:
            for letter_vector_num in range(test_data.size()[0]):
                output, hidden2 = self.rnn_module(test_data[letter_vector_num], hidden2)
            category_conf, category_index = output.topk(1)
            if category_index == category_idc:
                recall += 1
        
        return total_loss/len(self.training_list), recall/len(self.testing_list)
    
    def train_n_epochs(self, n_epochs = 10):
        losses = []
        recalls = []
        for i in range(n_epochs):
            loss, recall = self.train_one_epoch()
            print(f"Average training epoch loss after epoch {i+1} is {loss}; Recall of validation set: {recall}")
            losses.append(loss)
            recalls.append(recall)
        plt.plot(losses, label = "Training NLLLoss")
        plt.plot(recalls, label = "Validation Recall")
        plt.xlabel("Epochs")
        plt.legend(loc=1)
        plt.savefig("training_losses_validation_recall.png", dpi = 1000)

def main():
    classifier = RNNClassifier(learning_rate = 0.0005)
    classifier.set_data_dictionary()
    classifier.set_rnn_module()
    classifier.train_n_epochs(100)
    
if __name__ == "__main__":
    main()

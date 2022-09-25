import argparse
import os
import re
import math

from enum import Enum

class MessageType(Enum):
    REGULAR = 1,
    SPAM = 2

class Counter():

    def __init__(self):
        self.counter_regular = 0
        self.counter_spam = 0

    def increment_counter(self, message_type):
        """
        Increment a word's frequency count by one, depending on whether it occurred in a regular or spam message.
        :param message_type: The message type to be parsed (MessageType.REGULAR or MessageType.SPAM)
        :return: None
        """
        if message_type == MessageType.REGULAR:
            self.counter_regular += 1
        else:
            self.counter_spam += 1

class Bayespam():

    def __init__(self):
        self.regular_list = None
        self.spam_list = None
        self.vocab = {}

    def list_dirs(self, path):
        """
        Creates a list of both the regular and spam messages in the given file path.
        :param path: File path of the directory containing either the training or test set
        :return: None
        """
        # Check if the directory containing the data exists
        if not os.path.exists(path):
            print("Error: directory %s does not exist." % path)
            exit()

        regular_path = os.path.join(path, 'regular')
        spam_path = os.path.join(path, 'spam')

        # Create a list of the absolute file paths for each regular message
        # Throws an error if no directory named 'regular' exists in the data folder
        try:
            self.regular_list = [os.path.join(regular_path, msg) for msg in os.listdir(regular_path)]
        except FileNotFoundError:
            print("Error: directory %s should contain a folder named 'regular'." % path)
            exit()

        # Create a list of the absolute file paths for each spam message
        # Throws an error if no directory named 'spam' exists in the data folder
        try:
            self.spam_list = [os.path.join(spam_path, msg) for msg in os.listdir(spam_path)]
        except FileNotFoundError:
            print("Error: directory %s should contain a folder named 'spam'." % path)
            exit()

    def read_messages(self, message_type):
        """
        Parse all messages in either the 'regular' or 'spam' directory. Each token is stored in the vocabulary,
        together with a frequency count of its occurrences in both message types.
        :param message_type: The message type to be parsed (MessageType.REGULAR or MessageType.SPAM)
        :return: None
        """
        if message_type == MessageType.REGULAR:
            message_list = self.regular_list
        elif message_type == MessageType.SPAM:
            message_list = self.spam_list
        else:
            message_list = []
            print("Error: input parameter message_type should be MessageType.REGULAR or MessageType.SPAM")
            exit()

        for msg in message_list:
            try:
                # Make sure to use latin1 encoding, otherwise it will be unable to read some of the messages
                f = open(msg, 'r', encoding='latin1')

                # Loop through each line in the message
                for line in f:
                    # Split the string on the space character, resulting in a list of tokens
                    split_line = line.split(" ")
                    # Loop through the tokens
                    for idx in range(len(split_line)):
                        token = split_line[idx]
                        
                        ## Make the word lower case and remove any punctuation and numbers.
                        token = token.lower()
                        token = re.sub("[^a-zA-Z]+", "", token)
                        
                        ## Do not consider words with less than four letters.
                        if len(token) < 4:
                            continue
                        
                        if token in self.vocab.keys():
                            # If the token is already in the vocab, retrieve its counter
                            counter = self.vocab[token]
                        else:
                            # Else: initialize a new counter
                            counter = Counter()

                        # Increment the token's counter by one and store in the vocab
                        counter.increment_counter(message_type)
                        self.vocab[token] = counter
            except Exception as e:
                print("Error while reading message %s: " % msg, e)
                exit()

    def print_vocab(self):
        """
        Print each word in the vocabulary, plus the amount of times it occurs in regular and spam messages.
        :return: None
        """
        for word, counter in self.vocab.items():
            # repr(word) makes sure that special characters such as \t (tab) and \n (newline) are printed.
            print("%s | In regular: %d | In spam: %d" % (repr(word), counter.counter_regular, counter.counter_spam))

    def write_vocab(self, destination_fp, sort_by_freq=False):
        """
        Writes the current vocabulary to a separate .txt file for easier inspection.
        :param destination_fp: Destination file path of the vocabulary file
        :param sort_by_freq: Set to True to sort the vocab by total frequency (descending order)
        :return: None
        """

        if sort_by_freq:
            vocab = sorted(self.vocab.items(), key=lambda x: x[1].counter_regular + x[1].counter_spam, reverse=True)
            vocab = {x[0]: x[1] for x in vocab}
        else:
            vocab = self.vocab

        try:
            f = open(destination_fp, 'w', encoding="latin1")

            for word, counter in vocab.items():
                # repr(word) makes sure that special  characters such as \t (tab) and \n (newline) are printed.
                f.write("%s | In regular: %d | In spam: %d\n" % (repr(word), counter.counter_regular, counter.counter_spam),)

            f.close()
        except Exception as e:
            print("An error occurred while writing the vocab to a file: ", e)

def classifyMsg(msg, reg_lp, spam_lp, cond_prob_reg, cond_prob_spam):
    ## We left out the alpha from the posterior probabilities as we are not interested in the exact values, 
    ## but only how they compare to each other and as this is a constant factor we decide to leave it out, 
    ## as it does not have any impact on the comparison of the two class probabilities.
    reg_log_prob = reg_lp
    spam_log_prob = spam_lp
    try:

        f = open(msg, 'r', encoding='latin1')
        
        # Loop through each line in the message
        for line in f:
            # Split the string on the space character, resulting in a list of tokens
            split_line = line.split(" ")
            # Loop through the tokens
            for idx in range(len(split_line)):
                token = split_line[idx]
                
                ## Make the word lower case and remove any punctuation and numbers.
                token = token.lower()
                token = re.sub("[^a-zA-Z]+", "", token)
                        
                ## Do not consider words with less than four letters.
                if len(token) < 4:
                    continue
                if token in cond_prob_reg:
                    reg_log_prob += cond_prob_reg[token]
                    
                if token in cond_prob_spam:
                    spam_log_prob += cond_prob_spam[token]   
    except Exception as e:
        print("Error while reading message %s: " % msg, e)
        exit()
    
    if reg_log_prob > spam_log_prob:
        return MessageType.REGULAR
    else:
        return MessageType.SPAM


def classifyMsgs(bayespam_test, reg_lp, spam_lp, cond_prob_reg, cond_prob_spam):
    true_positive = 0
    false_negative = 0
    true_negative = 0
    false_positive = 0
    for msg in bayespam_test.regular_list:
        if classifyMsg(msg, reg_lp, spam_lp, cond_prob_reg, cond_prob_spam) == MessageType.REGULAR:
            true_positive += 1
        else:
            false_negative += 1
    
    for msg in bayespam_test.spam_list:
        if classifyMsg(msg, reg_lp, spam_lp, cond_prob_reg, cond_prob_spam) == MessageType.SPAM:
            true_negative += 1
        else:
            false_positive += 1
    return true_positive, false_negative, false_positive, true_negative
    
def main():
    # We require the file paths of the training and test sets as input arguments (in that order)
    # The argparse library helps us cleanly parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('train_path', type=str,
                        help='File path of the directory containing the training data')
    parser.add_argument('test_path', type=str,
                        help='File path of the directory containing the test data')
    args = parser.parse_args()

    # Read the file path of the folder containing the training set from the input arguments
    train_path = args.train_path

    # Initialize a Bayespam object
    bayespam = Bayespam()
    # Initialize a list of the regular and spam message locations in the training folder
    bayespam.list_dirs(train_path)

    # Parse the messages in the regular message directory
    bayespam.read_messages(MessageType.REGULAR)
    # Parse the messages in the spam message directory
    bayespam.read_messages(MessageType.SPAM)

    # bayespam.print_vocab()
    # bayespam.write_vocab("vocab.txt")

    print("N regular messages: ", len(bayespam.regular_list))
    print("N spam messages: ", len(bayespam.spam_list))
    
    total_num = len(bayespam.regular_list) + len(bayespam.spam_list)
    
    ## Determine the a priori probabilities of the classes
    prior_prob_reg = math.log(len(bayespam.regular_list) / total_num )
    prior_prob_spam = math.log(len(bayespam.spam_list) / total_num )
    
    n_word_regular = 0
    n_word_spam = 0
    for word, counter in bayespam.vocab.items():
        n_word_regular += counter.counter_regular
        n_word_spam += counter.counter_spam
    
    cond_prob_reg = {}
    cond_prob_spam = {}
    e = 0.5
    for word, counter in bayespam.vocab.items():
        if counter.counter_regular == 0:
            cond_prob_reg[word] = math.log(e / (n_word_regular + n_word_spam) )
        else:
            cond_prob_reg[word] = math.log(counter.counter_regular / n_word_regular)
        if counter.counter_spam == 0:
            cond_prob_spam[word] = math.log(e / (n_word_regular + n_word_spam) )
        else:
            cond_prob_spam[word] = math.log(counter.counter_spam / n_word_spam)
    
    bayespam_test = Bayespam()
    
    bayespam_test.list_dirs(args.test_path)
    
    true_positive, false_negative, false_positive, true_negative = classifyMsgs(bayespam_test, prior_prob_reg, prior_prob_spam, cond_prob_reg, cond_prob_spam)
    
    print("Confusion Matrix:\n%d, %d\n%d, %d\n\nTrue Positive Rate:\n%f\nTrue Negative Rate:\n%f"% (true_positive, false_positive, false_negative, true_negative, float(true_positive)/float(true_positive + false_negative), float(true_negative)/float(true_negative + false_positive)))
    
    """
    Now, implement the follow code yourselves:
    1) A priori class probabilities must be computed from the number of regular and spam messages
    2) The vocabulary must be clean: punctuation and digits must be removed, case insensitive
    3) Conditional probabilities must be computed for every word
    4) Zero probabilities must be replaced by a small estimated value
    5) Bayes rule must be applied on new messages, followed by argmax classification
    6) Errors must be computed on the test set (FAR = false accept rate (misses), FRR = false reject rate (false alarms))
    7) Improve the code and the performance (speed, accuracy)
    
    Use the same steps to create a class BigramBayespam which implements a classifier using a vocabulary consisting of bigrams
    """

if __name__ == "__main__":
    main()
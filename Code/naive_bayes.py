import re
from typing import Set,Dict,List,Tuple
from collections import defaultdict
class Message:
    text:str
    is_spam = bool
def takeword(text:str)->Set[str]:
    text = text.lower()
    all_word = re.findall("[a-z0-9]+",text)
    return set(all_word)

class NaiveBayesClassifier:
    def __init__(self,k:float=0.5)->None:
        self.k = k
        self.tokens = Set[str] = set()
        self.token_spam_counts:Dict[str,int]=defaultdict(int)
        self.token_ham_counts:Dict[str,int]=defaultdict(int)
        self.spam_messages = self.ham_message=0
    def train(self,messages:List[Message]):
        for message in messages:
            if message.is_spam:
                self.spam_messages +=1
            else:
                self.ham_message +=1
            for word in takeword(message):
                self.tokens.add(word)
                if message.is_spam:
                    self.token_spam_counts[word] +=1
                else:
                    self.token_ham_counts[word] +=1
    def _probabilities(self,token:str)->Tuple[float,float]:
        spam = self.token_spam_counts[token]
        ham = self.token_ham_counts[token]
        p_token_spam = (self.k+spam)/(self.k*2+self.spam_messages)
        p_token_ham = (self.k+ham)/(self.k*2+self.ham_message)
        return p_token_ham,p_token_spam




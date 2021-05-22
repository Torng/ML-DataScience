import re
from typing import Set,Dict,List,Tuple,NamedTuple
from collections import defaultdict
import math
class Message(NamedTuple):
    text:str
    is_spam:bool
def takeword(text:str)->Set[str]:
    text = text.lower()
    all_word = re.findall("[a-z0-9]+",text)
    return set(all_word)

class NaiveBayesClassifier:
    def __init__(self,k:float=0.5)->None:
        self.k = k
        self.tokens :Set[str] = set()
        self.token_spam_counts:Dict[str,int]=defaultdict(int)
        self.token_ham_counts:Dict[str,int]=defaultdict(int)
        self.spam_messages = self.ham_message=0
    def train(self,messages:List[Message]):
        for message in messages:
            if message.is_spam:
                self.spam_messages += 1
            else:
                self.ham_message += 1
            for word in takeword(message.text):
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
    def predict(self,text:str)->float:
        text_token = takeword(text)
        log_prob_if_spam = log_prob_if_ham = 0
        for toekn in self.tokens:
            prob_is_spam,prob_is_ham = self._probabilities(toekn)

            if(toekn in text_token):
                log_prob_if_spam += math.log(prob_is_spam)
                log_prob_if_ham += math.log(prob_is_ham)
            else:
                log_prob_if_spam += math.log(1.0-prob_is_spam)
                log_prob_if_ham += math.log(1.0-prob_is_ham)
        prob_is_ham =  math.exp(log_prob_if_ham)
        prob_is_spam = math.exp(log_prob_if_spam)
        return prob_is_spam/(prob_is_spam+prob_is_ham)

messages = [Message("price",True),
        Message("Hawk",False),
        Message("jenny",False)
           ]
model = NaiveBayesClassifier()
model.train(messages)

print(model.predict("price = 10000"))




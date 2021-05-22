import re
from bs4 import BeautifulSoup
import requests
from collections import defaultdict
from typing import DefaultDict,Dict,List
import random
Grammar = Dict[str,List[str]]
def generate_using_bigrams(transitions:DefaultDict)-> str:
    current = "."
    result = []
    while True:
        next_word_candidates = transitions[current]
        current = random.choice(next_word_candidates)
        result.append(current)
        if current == "." :
            return " ".join(result)
def fix_unicode(txet:str):
    return txet.replace(u"\u2019","'")

def is_terminal(token:str)->bool:
    return token[0] != "_"

def expand(grammar:Grammar,tokens:List[str])->List[str]:
    for i , token in enumerate(tokens):
        if is_terminal(token):
            continue
        replacement = random.choice(grammar[token])

        if is_terminal(replacement):
            tokens[i] = replacement
        else:
            tokens = tokens[:i]+replacement.split()+tokens[(i+1):]
        return tokens
def generate_sentence(grammer:Grammar)->List[str]:
    return expand(grammar,["_S"])

#Get Web text start
url = "https://www.managertoday.com.tw/articles/view/62614"
html = requests.get(url).text
soup = BeautifulSoup(html,'lxml')
content = soup.find("div","PostHtmlView d-md-flex")
regex = r"[\w']+|[\.]"
document = []
for p in content('p'):
    words = re.findall(regex,p.text)
    document.extend(words)
#Get Web text End

transitions = defaultdict(list)
# 2-gram *
for prev,current in zip(document,document[1:]):
    transitions[prev].append(current)
# 2-gram &
print(generate_using_bigrams(transitions))
# print()


grammar = {
    "_S"  : ["_NP _VP"],
    "_NP" : ["_N",
             "_A _NP _P _A _N"],
    "_VP" : ["_V",
             "_V _NP"],
    "_N"  : ["data science", "Python", "regression"],
    "_A"  : ["big", "linear", "logistic"],
    "_P"  : ["about", "near"],
    "_V"  : ["learns", "trains", "tests", "is"]
}
# print(generate_sentence(grammar))







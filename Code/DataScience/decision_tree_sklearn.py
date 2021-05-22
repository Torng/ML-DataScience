from sklearn import tree
# Senior = 0 ,Junior = 1, Mid = 2 
# Python = 0 ,Java = 1 , R = 2
datas = [ 
        (0,1,False,False,   False), 
        (0,1,False,True,  False),
        (2,0,False,False,     True),
        (1,0,False,False,  True),
        (1,2,True,False,      True),
        (1,2,True,True,    False),
        (2,2,True,True,        True),
        (0,0,False,False, False),
        (0,2,True,False,      True),
        (1,0,True,False, True),
        (0,0,True,True,True),
        (2,0,False,True,    True),
        (2,1,True,False,      True),
        (1,0,False,True,False)
    ]
train_x = [[data[0],data[1],data[2],data[3]]for data in datas]
train_y = [data[4]for data in datas]
clf = tree.DecisionTreeClassifier()
clf.fit(train_x,train_y)

print(clf.predict([[0,1,False,False]]))

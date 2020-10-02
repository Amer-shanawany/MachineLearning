#!/usr/bin/python


try:
    # a,b,c,d,e = input("Enter five values: ").split()
    # for multiple values
    x = list(map(int, input("Enter a multiple value seperated by a space: ").split()))
    x.sort()
    print("Multiple numbers sorted list: ", x)
    ##### 
    # for five values only 
    y=[]
    for i in range(5):
        y.append( int(input("number "+str(i+1)+" : ")))
    y.sort()
    print("Five numbers sorted list: ", y)
except Exception as Errors:
    print(Errors)

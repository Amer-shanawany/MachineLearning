#!/usr/bin/python

menu = """ 
        1) Enter a filename
        2) Print the content of a file
        3) Edit the file

        0) to Exit
"""



while True:

    try:
        print(menu)
        menuButton = int(input("Enter a command number: "))
        if menuButton == 0:
            print('Bye Bye!')
            break
        if menuButton == 1:
            fileName = input("Enter the file path and name: ")
        if menuButton == 2:
            f = open(fileName,'r+')# r+ open file for both reading and writing 
            dataFile = f.read()
            print(dataFile)
        if menuButton == 3:
            data2Write = input("Enter your text to be added to the file: \n\n")
            save2File = input("Do you want to append it to the file: [Y/n]")
            if save2File == 'Y' or save2File =='y':
                f.write(data2Write)
                print("you choose Y")
                f.close()
                print("File is saved")
    except Exception as Errors:
        print(Errors)

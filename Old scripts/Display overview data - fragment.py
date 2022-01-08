import tkinter as tk
from tkinter import ttk
from tkinter import *

# this is the function called when the button is clicked
def datePrev():
	print('clicked')


# this is the function called when the button is clicked
def dateNext():
	print('clicked')


# this is a function to get the user input from the text input box
def getInputBoxValue():
	userInput = dateInput.get()
	return userInput


# this is a function to get the selected list box value
def getListboxValue():
	itemSelected = listBoxOne.curselection()
	return itemSelected



root = Tk()

# This is the section of code which creates the main window
root.geometry('580x350')
root.configure(background='#FFFFFF')
root.title('Main window')


# This is the section of code which creates a button
Button(root, text='Back', bg='#999999', font=('arial', 12, 'normal'), command=datePrev).place(x=99, y=33)


# This is the section of code which creates a button
Button(root, text='Next', bg='#999999', font=('arial', 12, 'normal'), command=dateNext).place(x=269, y=33)


# This is the section of code which creates a text input box
dateInput=Entry(root)
dateInput.place(x=169, y=43)


# This is the section of code which creates the a label
Label(root, text='Date', bg='#FFFFFF', font=('arial', 12, 'normal')).place(x=39, y=43)


# This is the section of code which creates the a label
Label(root, text='Supplements - overview', bg='#FFFFFF', font=('arial', 12, 'normal')).place(x=39, y=133)


# This is the section of code which creates a listbox
listBoxOne=Listbox(root, bg='#999999', font=('arial', 12, 'normal'), width=0, height=0)
listBoxOne.insert('0', 'hot dogs')
listBoxOne.insert('1', 'curry')
listBoxOne.insert('2', 'falafel')
listBoxOne.insert('3', 'Reuben')
listBoxOne.insert('4', 'chocolate')
listBoxOne.place(x=269, y=143)


root.mainloop()

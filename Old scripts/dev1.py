import numpy as np
import tkinter as tk

def dev2():

    global root, ws, hs, tkgui
    root = tk.Tk()
    ws, hs = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry('%dx%d+%d+%d' % (ws, hs, 0, 0))  # width, height, x, y, full screen display here now.

    class tkclass:
        pass
    tkgui = tkclass

    l = int(ws / 3)
    pad = 20
    s0 = ''.join([str(a) for a in np.arange(300)])
    s1 = ''.join([str(a) for a in np.arange(300, 600, 1)])
    s2 = ''.join([str(a) for a in np.arange(600, 900, 1)])
    str0 = tk.StringVar()
    str1 = tk.StringVar()
    str2 = tk.StringVar()
    str0.set(s0)
    str1.set(s1)
    str2.set(s2)

    tkgui.l0 = tk.Label(root, textvariable=str0, anchor='nw', wraplength=l)
    tkgui.l0.grid(row=0, rowspan=2, column=0, ipadx=pad, sticky='n')
    tkgui.l01 = tk.Label(root, textvariable=str0, anchor='sw', wraplength=l)
    tkgui.l01.grid(row=1, rowspan=1, column=0, ipadx=pad, sticky='s')
    tkgui.l1 = tk.Label(root, textvariable=str1, anchor='nw', wraplength=l)
    tkgui.l1.grid(row=0, rowspan=3, column=1, ipadx=pad, sticky='n')
    tkgui.l2 = tk.Label(root, textvariable=str2, anchor='nw', wraplength=l)
    tkgui.l2.grid(row=0, rowspan=3, column=2, ipadx=pad, sticky='n')

    # tkgui.l0 = tk.Label(root, text=str0, anchor='nw', wraplength=l)
    # tkgui.l0.grid(row=0, column=0, ipadx=pad)  #, sticky='NSEW')
    # tkgui.l1 = tk.Label(root, text=str1, anchor='nw', wraplength=l)
    # tkgui.l1.grid(row=0, column=1, ipadx=pad)  # , sticky='NSEW')
    # tkgui.l2 = tk.Label(root, text=str2, anchor='nw', wraplength=l)
    # tkgui.l2.grid(row=0, column=2, ipadx=pad)  #, sticky='NSEW')

    root.rowconfigure(0, weight=1)
    root.rowconfigure(1, weight=1)
    root.rowconfigure(2, weight=1)
    root.columnconfigure(0, weight=1)
    root.columnconfigure(1, weight=1)
    root.columnconfigure(2, weight=1)
    root.mainloop()

dev2()
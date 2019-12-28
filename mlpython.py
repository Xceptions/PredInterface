import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
from tkinter import *
import pandas as pd
from sklearn.preprocessing import StandardScaler

from sklearn.externals import joblib
import numpy as np
import pandas as pd

class APP():
    def __init__(self):
        self.win = tk.Tk()
        self.win.title('WAT Predictions Using Adam Optimizer')
        self.win.geometry('1000x500')
        self.createTabs()
        self.addTabElements()

    def _upload(self):
        messagebox.showinfo('info', 'Predicting...')
        self.win.configure(background="pink")

    def createTabs(self):
        tabControl = ttk.Notebook(self.win)
        self.predictTab = ttk.Frame(tabControl)
        tabControl.add(self.predictTab, text="Predict")
        tabControl.pack(expand=1, fill="both")

    def addTabElements(self):
        self.header = tk.Label(
            self.predictTab,
            text='Upload your CSV File',
            font="Times 40",
            width=25,
            height=4
        )
        self.header.grid(row=250, column=500)
        self.button = tk.Button(
            self.predictTab,
            text='upload',
            command=self.UploadAction,
            width=25,
            height=4,
            bd=1
        )
        self.button.grid(column=500, row=251)


    def UploadAction(self):
        filename = filedialog.askopenfilename(initialdir = "./",title = "Select file",filetypes = (("csv files","*.csv"),("all files","*.*")))
        df = pd.read_csv(filename)
        OIL_SAMPLE_NUMBER = df['OIL_SAMPLE_NUMBER'].copy()
        df = df.drop(['OIL_SAMPLE_NUMBER'], axis=1)

        loaded_model = joblib.load('finalized_model.sav')

        density = df['DENSITY'].copy()
        molecular_weight = df['MOLECULAR_WEIGHT'].copy()
        pressure = [14.504 for x in density]
        AE = df['ACTIVATION_ENERGY'].copy()
        enthalpy = df['ENTHALPY_CHANGE'].copy()
        kinematic_visc = df['KINEMATIC_VISCOSITY'].copy()

        temp = [80 for x in density]
        df['R_constant'] = (density * temp) / (pressure * molecular_weight) # generated feature
        # R_const = df['R_constant'].copy()
        df['dynamic_visc'] = kinematic_visc * density # generated feature
        df['another_visc'] = kinematic_visc * molecular_weight
        df['density_per_pressure'] = pressure / density
        dynamic_visc = df['dynamic_visc'].copy()
        kinematic_visc = df['KINEMATIC_VISCOSITY'].copy()
        df['pressure_per_dynamic_visc'] = pressure / dynamic_visc
        df['pressure_per_kinematic_visc'] = pressure / kinematic_visc
        
        data = df.values
        sc = StandardScaler()

        data = sc.fit_transform(data)
        predictions = loaded_model.predict(data)
        filename = 'result.csv'
        new_df = pd.DataFrame({})
        new_df['OIL_SAMPLE_NUMBER'] = OIL_SAMPLE_NUMBER
        new_df['WAT'] = predictions
        new_df.to_csv(filename, index=False)

        messagebox.showinfo('info', 'Successful! predictions saved as: result.csv')


app = APP()
app.win.mainloop()
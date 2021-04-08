# %%

# Copyright Pihda Di Ba 2020
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import re
import ast
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

# %%

# Helper class go retreve data data dari worldometers.info

regression = {}

# X and Y axis for comparison
global_graph_x = []
global_graph_y = []


class Coronavirus():
    def __init__(self, country: str, end_date: str = "", slice_elem: bool = False):
        self.slice_elem = slice_elem
        self.country = country
        options = Options()
        options.headless = True
        self.driver = webdriver.Chrome(options=options)
        self.xc: np.array = np.empty(0)
        self.yc: np.array = np.empty(0)
        self.xc2: np.array = np.empty(0)
        self.yc2: np.array = np.empty(0)
        self.xd: np.array = np.empty(0)
        self.yd: np.array = np.empty(0)
        self.xint: np.array = np.empty(0)
        self.reg_arr = []
        self.end_date = end_date
        self.driver.get('https://www.worldometers.info/coronavirus/country/' + self.country)
        self.r2 = ""

    def __del__(self):
        self.driver.close()
        self.driver.quit()

    def c_get_data(self):
        self.cases_data()

    def c_lin_reg(self, asc: bool):
        if asc:
            self.linear_regression('c', self.xc, self.yc, asc)
            print("Linier regression formula for " + self.country + " is :")
            print(regression[self.country + "_c_lin_reg_asc"])

        else:
            self.linear_regression('c', self.xc2, self.yc2, asc)
            print("Linier regression formula (decending) active cases for " + self.country + " is :")
            print(regression[self.country + "_c_lin_reg_dec"])

    def c_quad_reg(self, asc: bool):
        if asc:
            self.quadratic_regression('c', self.xc, self.yc, asc)
            print("Persamaan regresi kuadrat kasus aktif di negara " + self.country + " adalah :")
            print(regression[self.country + "_c_quad_reg_asc"])


        else:
            self.quadratic_regression('c', self.xc2, self.yc2, asc)
            print("Persamaan regresi (grafik turun) kuadrat kasus aktif di negara " + self.country + " adalah :")
            print(regression[self.country + "_c_quad_reg_dec"])

    def c_exp_reg(self, asc: bool):

        if asc:
            self.exponential_regression('c', self.xc, self.yc, asc)
            print("Persamaan regresi exponen kasus aktif di negara " + self.country + " adalah :")
            print(regression[self.country + "_c_exp_reg_asc"])

        else:
            self.exponential_regression('c', self.xc2, self.yc2, asc)
            print("Persamaan regresi (grafik turun) exponen kasus aktif di negara " + self.country + " adalah :")
            print(regression[self.country + "_c_exp_reg_dec"])

    def c_ply_reg(self, asc: bool, iter: int, num: int):

        if asc:
            self.polynomial_regression('c', self.xc, self.yc, asc, iter, num)
            print("Persamaan regresi polinomial kasus aktif di negara " + self.country + " adalah :")
            print(regression[self.country + "_c_ply_reg_asc"])
        else:
            self.polynomial_regression('c', self.xc2, self.yc2, asc, iter, num)
            print("Persamaan regresi (grafik turun) polinomial kasus aktif di negara " + self.country + " adalah :")
            print(regression[self.country + "_c_ply_reg_dec"])

    def c_ply_reg_lim(self, asc: bool, iter: int, num: int, data_len: int):
        if data_len == 0:
            xc = self.xc
            yc = self.yc
        else:
            xc = self.xc[:data_len]
            yc = self.yc[:data_len]
        self.polynomial_regression('c', xc, yc, asc, iter, num, True)

    def d_get_data(self):
        self.death_data()

    def d_lin_reg(self):
        self.linear_regression('d', self.xd, self.yd, True)
        print("Persamaan regresi linier jumlah kematian di negara " + self.country + " adalah:")
        print("Dengan R2 : " + str(self.r2))

    def d_quad_reg(self):
        self.quadratic_regression('d', self.xd, self.yd, True)
        print("Persamaan regresi kuadrat kematian di negara " + self.country + " adalah :")
        print(regression[self.country + "_d_quad_reg_asc"])

    def d_exp_reg(self):
        self.exponential_regression('d', self.xd, self.yd, True)
        print("Persamaan regresi exponen jumlah kematian di negara " + self.country + " adalah :")
        print(regression[self.country + "_d_exp_reg_asc"])

    def d_ply_reg(self, iter: int, num: int):
        self.polynomial_regression('d', self.xd, self.yd, True, iter, num)
        print("Persamaan regresi polinomial jumlah kematian di negara " + self.country + " adalah :")
        print(regression[self.country + "_d_ply_reg_asc"])

    ### Retreves and converts the javascript in worldometers.info

    def parse_data(self, what):
        js_obj = what.find_element_by_xpath('../..').find_element_by_tag_name('script').get_attribute('innerHTML')
        x_raw = re.findall('(.*)categories(.*)', js_obj)
        y_raw = re.findall('(.*)data(.*)', js_obj)

        x = np.array(ast.literal_eval(re.findall('\[.*?\]', x_raw[0][1])[0]))
        y = np.array(ast.literal_eval(re.findall('\[.*?\]', y_raw[0][1])[0]))
        return x, y

    ### Retreve Covid-19 from id graph-active-cases-total

    def cases_data(self):
        peaks = 0
        table_deaths = self.driver.find_element_by_id('graph-active-cases-total')
        self.xc, self.yc = self.parse_data(table_deaths)

        # Eleminate data with zero value
        self.yc = self.yc[self.yc != 0]
        self.xc = self.xc[-len(self.yc):]

        if self.end_date != "":
            idx = int(np.where(self.xc == self.end_date)[0])
            self.xc = self.xc[:idx + 1]
            self.yc = self.yc[:idx + 1]

        if self.slice_elem:
            peaks, _ = find_peaks(self.yc, height=0)

        fig = plt.figure(figsize=(18, 16), dpi=80, facecolor='w', edgecolor='k')
        ax = plt.plot(self.xc, self.yc, 'ro')

        if self.slice_elem:
            plt.plot(peaks, self.yc[peaks], "x", markersize=22)

        plt.grid()
        locs, labels = plt.xticks()
        plt.setp(labels, rotation=90)

        plt.xlabel('Date')
        plt.ylabel('Cases')

        if self.slice_elem:
            max_peak_idx = np.where(self.yc == np.max(self.yc))[0][0]
            self.xc2 = self.xc[max_peak_idx:]
            self.yc2 = self.yc[max_peak_idx:]

            self.xc = self.xc[:max_peak_idx + 1]
            self.yc = self.yc[:max_peak_idx + 1]

    ### Retreve death number from the tabbable-panel-deaths css class

    def death_data(self):
        table_deaths = self.driver.find_element_by_class_name('tabbable-panel-deaths')
        self.xd, self.yd = self.parse_data(table_deaths)

        # Eleminate data with zero value
        self.yd = self.yd[self.yd != 0]

        self.xd = self.xd[-len(self.yd):]

        if self.end_date != "":
            idx = int(np.where(self.xd == self.end_date)[0])
            self.xd = self.xd[:idx + 1]
            self.yd = self.yd[:idx + 1]

        fig = plt.figure(figsize=(18, 16), dpi=80, facecolor='w', edgecolor='k')
        ax = plt.plot(self.xd, self.yd, 'ro')
        plt.title("Jumlah kematian akibat Covid-19 di " + self.country)

        plt.grid()
        locs, labels = plt.xticks()
        plt.setp(labels, rotation=90)

        plt.xlabel('Date')
        plt.ylabel('Death(s)')

    def linear_regression(self, data_type: str, x: np.array, y: np.array, asc: bool):
        x_date = x
        x = np.arange(1, len(x) + 1, 1)
        n = len(y)
        xy: np.array = np.array(x * y)
        xx: np.array = np.power(x, 2)

        B = (n * xy.sum() - x.sum() * y.sum()) / (n * xx.sum() - (x.sum()) ** 2)

        A = y.mean() - B * x.mean()

        ### Calculate Regression coefficient

        yDt = (y - y.mean()) ** 2
        yD = (y - A - B * y) ** 2

        r = np.sqrt(abs((yDt.sum() - yD.sum()) / yDt.sum()))

        self.r2 = r ** 2

        yreg = B * x + A

        regression.update({self.country + "_" + data_type + "_lin_reg_" + ("asc" if asc else "dec"): [
            str(f"y = {B:.4f}x + {A:.4f}"), "R2 = " + str(self.r2)]})

        fig = plt.figure(figsize=(18, 16), dpi=80, facecolor='w', edgecolor='k')
        ax = plt.plot(x_date, y, 'ro')
        plt.plot(x_date, yreg, 'b')
        plt.grid()
        locs, labels = plt.xticks()
        plt.setp(labels, rotation=90)
        plt.title("Regresi linier untuk grafik " + ('naik' if asc else 'turun'))
        plt.xlabel('Hari ke')
        plt.ylabel('Jumlah')

    def quadratic_regression(self, data_type: str, x: np.array, y: np.array, asc: bool):
        x_date = x
        x = np.arange(1, len(x) + 1, 1)
        p = np.log10(y)
        q = np.log10(x)

        n = len(q)
        pq = p * q
        qq = np.power(q, 2, out=np.zeros_like(q), where=q != 0)

        B = (n * pq.sum() - q.sum() * p.sum()) / (n * qq.sum() - (q.sum()) ** 2)

        A = p.mean() - B * q.mean()

        print("p = {:.4f}q+{:.4f}".format(B, A))

        a = 10 ** A  # A=log(a)
        b = B

        ### Calculate Regression coefficient
        yDt = (y - y.mean()) ** 2
        yD = (y - a * (x ** b)) ** 2

        r = np.sqrt(abs((yDt.sum() - yD.sum()) / yDt.sum()))
        self.r2 = r ** 2

        ### Display grafic

        xreg = np.array(range(0, int((x[-1] + 1) * 10)))
        xreg = xreg / 10
        if asc:
            yreg = a * (xreg ** b)
        else:
            xreg = xreg[1:]
            yreg = a * (xreg ** b)
        # Store data in a Dictionary
        regression.update({self.country + "_" + data_type + "_quad_reg_" + ("asc" if asc else "dec"): [
            "y = {:.4f}x^({:.4f})".format(a, b), "R2 = " + str(self.r2)]})

        print(len(x_date))
        print(len(y))

        fig = plt.figure(figsize=(18, 16), dpi=80, facecolor='w', edgecolor='k')
        ax = plt.plot(x_date, y, 'ro')
        plt.plot(xreg, yreg, 'b')
        plt.grid()
        locs, labels = plt.xticks()
        plt.setp(labels, rotation=90)
        plt.title("Regresi pangkat untuk grafik " + ('naik' if asc else 'turun'))
        plt.xlabel('Hari ke')
        plt.ylabel('Jumlah')

    def exponential_regression(self, data_type: str, x: np.array, y: np.array, asc: bool):
        x_date = x
        x = np.arange(1, len(x) + 1, 1)
        p = np.log(y)
        q = x

        n = len(q)
        pq = p * q
        qq = q ** 2

        B = (n * pq.sum() - q.sum() * p.sum()) / (n * qq.sum() - (q.sum()) ** 2)
        A = p.mean() - B * q.mean()

        a = np.e ** A  # A=log(a)
        b = B

        ### R2
        yDt = (y - y.mean()) ** 2
        yD = (y - a * (np.e ** (B * x))) ** 2

        r = np.sqrt(abs((yDt.sum() - yD.sum()) / yDt.sum()))
        self.r2 = r ** 2

        regression.update({self.country + "_" + data_type + "_exp_reg_" + ("asc" if asc else "dec"): [
            "y = {:.4f}e^({:.4f}x)".format(a, b), "R2 = " + str(self.r2)]})

        xreg = np.array(range(0, int((x[-1] + 1) * 10)))
        xreg = xreg / 10
        yreg = a * (np.e ** (B * xreg))

        fig = plt.figure(figsize=(18, 16), dpi=80, facecolor='w', edgecolor='k')

        ax = plt.plot(x_date, y, 'ro')
        plt.plot(xreg, yreg, 'b')
        plt.grid()
        locs, labels = plt.xticks()
        plt.setp(labels, rotation=90)
        plt.title("Exponential regression " + ('ascending' if asc else 'decending'))
        plt.xlabel('Day')
        plt.ylabel('Total')

    def polynomial_regression(self, data_type: str, x: np.array, y: np.array, asc: bool, iter: int = 100, num: int = 5,
                              lim=False):
        # iter = num of  gauss seidel iterations
        # num = num of variabel
        x_date = x
        x = np.arange(1, len(x) + 1, 1)

        mat_a = []
        mat_b = []
        g = []

        for i in range(num):
            g.append(0)
            arr = []
            for j in range(num):
                if j == 0 and i == 0:
                    arr.append(len(x))
                else:
                    arr.append(np.power(x.tolist(), (j + i)).sum())

            mat_a.append(arr)
            mat_b.append((np.power(x.tolist(), i) * y.tolist()).sum())

        for i in range(0, iter):
            g = self.seidel(mat_a, g, mat_b)

        xreg = np.array(range(0, int((x[-1] + 1) * 10)))
        xreg = xreg / 10

        func = ""
        for i in range(num):
            if i == 0:
                yreg = g[i]
                func += f"y = {g[i]:.4}"
            else:
                yreg = yreg + g[i] * (xreg ** i)
                func += f" + {g[i]:.4} * (x ** {i})"

        # R2
        for i in range(num):
            if i == 0:
                yr = g[i]
            else:
                yr = yr + g[i] * (x ** i)
        yDt = (y - y.mean()) ** 2
        yD = (y - yr) ** 2
        r = np.sqrt(abs((yDt.sum() - yD.sum()) / yDt.sum()))
        self.r2 = r ** 2

        regression.update(
            {self.country + "_" + data_type + "_ply_reg_" + ("asc" if asc else "dec"): [func, "R2 = " + str(self.r2)]})

        if lim:
            global_graph_x.append(xreg)
            global_graph_y.append(yreg)
            return

        # Display graphic
        fig = plt.figure(figsize=(18, 16), dpi=80, facecolor='w', edgecolor='k')
        ax = plt.plot(x_date, y, 'ro')
        plt.plot(xreg, yreg, 'b')

        plt.grid()
        locs, labels = plt.xticks()
        plt.setp(labels, rotation=90)
        plt.title("Polynomial regression with " + str(num) + " variable " + ('ascending' if asc else 'decending'))
        plt.xlabel('Day')
        plt.ylabel('Total')

    def seidel(self, k: np.ndarray, l, m: np.ndarray):
        n = len(k)
        for j in range(0, n):
            d = m[j]
            for i in range(0, n):
                if j != i:
                    d -= k[j][i] * l[i]
            l[j] = d / k[j][j]
        return l


# %% md

## Regression for China

# %%

# china
bot = Coronavirus('china', 'Mar 26', slice_elem=True)
bot.c_get_data()

# %%

bot.c_lin_reg(True)

# %%

bot.c_lin_reg(False)

# %%

bot.c_quad_reg(True)

# %%

bot.c_quad_reg(False)

# %%

bot.c_exp_reg(True)

# %%

bot.c_exp_reg(False)

# %%

bot.c_ply_reg(True, 500, 6)

# %%

bot.c_ply_reg(False, 500, 5)

# %%

bot.d_get_data()

# %%

bot.d_lin_reg()

# %%

bot.d_quad_reg()

# %%

bot.d_exp_reg()

# %%

bot.d_ply_reg(500, 5)

# %%

del bot

# %% md

## Regression for USA

# %%

# no date limit
bot = Coronavirus('us', '', slice_elem=False)
bot.c_get_data()

# %%

bot.c_lin_reg(True)

# %%

bot.c_quad_reg(True)

# %%

bot.c_exp_reg(True)

# %%

bot.c_ply_reg(True, 500, 6)

# %%

bot.d_get_data()

# %%

bot.d_lin_reg()

# %%

bot.d_quad_reg()

# %%

bot.d_exp_reg()

# %%

bot.d_ply_reg(500, 6)

# %%

bot.c_ply_reg_lim(True, 500, 6, 25)

# %%

del bot

# %% md

## All the regression formula

# %%

regression

# %%

# for comparison
'''
print("")
print("")
print("")
fig = plt.figure(figsize=(18, 16), dpi=80, facecolor='w', edgecolor='k')
for i, xreg in enumerate(global_graph_x):
    if i == 0:
        plt.plot(xreg, global_graph_y[i], 'b')
    elif i == 1:
        plt.plot(xreg, global_graph_y[i], 'r')

plt.grid()
locs, labels = plt.xticks()
plt.setp(labels, rotation=90)
plt.title("Compare")
plt.xlabel('Day')
plt.ylabel('Total')
# regression'''

# %%





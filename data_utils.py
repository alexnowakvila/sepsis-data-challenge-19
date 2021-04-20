import numpy as np, os, sys
import pdb
from pandas import DataFrame
import matplotlib.pyplot as plt
import pandas as pd

""" Model class """

# def load_data(self, input_directories):
#     for input_directory in input_directories:

class Data(object):
    def __init__(self):
        # load data
        self.num_features = 40
        self.num_files = None
        self.lengths = None
        self.labels = None
        self.data = None
        self.data_matrix = None
        self.labels_array = None
        self.files = None
        self.column_names = None
        self.stats = None
        self.sepsis_prop = None

    def _load_data(self, file):
        with open(file, 'r') as f:
            header = f.readline().strip()
            column_names = header.split('|')
            data = np.loadtxt(f, delimiter='|')

        if column_names[-1] == 'SepsisLabel':
            self.column_names = column_names
            column_names = column_names[:-1]
            datay = data[:, -1]
            data = data[:, :-1]
        else:
            raise ValueError("There is no sepsislabel")
        return data, datay

    def load_data(self, input_directories, top=None):
        # Create list of patient files
        # create a big data matrix
        self.data = []
        self.labels = []
        self.lengths = []
        
        for input_directory in input_directories:
            files = []
            for f in os.listdir(input_directory):
                if os.path.isfile(os.path.join(input_directory, f)) and \
                    not f.lower().startswith('.') and f.lower().endswith('psv'):
                    files.append(f)
            if top: files = files[:top]  # use a subset of files
            num_files = len(files)
        
            for i, f in enumerate(files):
                if i % 1000 == 0: print('    {}/{}...'.format(i, num_files))
                # Load data.
                input_file = os.path.join(input_directory, f)
                data_unit, datay_unit = self._load_data(input_file)
                length = data_unit.shape[0]
                self.data.append(data_unit)
                self.labels.append(datay_unit)
                self.lengths.append(length)
        self.num_files = len(self.lengths)
        self.data_matrix = np.concatenate(self.data, axis=0)  # concat data
        self.labels_array = np.concatenate(self.labels, axis=0)

    def basic_statistics(self):

        mean = np.nanmean(self.data_matrix, axis=0)
        var = np.nanvar(self.data_matrix, axis=0)
        nanmax = np.nanmax(self.data_matrix, axis=0)
        nanmin = np.nanmin(self.data_matrix, axis=0)
        nancount = np.sum(np.isnan(self.data_matrix).astype(int), 0)
        nanprop = (nancount).astype(float) / self.data_matrix.shape[0]
        nanper25 = np.nanpercentile(self.data_matrix, q=25, axis=0)
        nanper50 = np.nanpercentile(self.data_matrix, q=50, axis=0)
        nanper75 = np.nanpercentile(self.data_matrix, q=75, axis=0)

        df = pd.DataFrame(data=mean, index=self.column_names[:-1], \
            columns=["Mean"])
        df["Min"] = nanmin
        df["Per 0.25"] = nanper25
        df["Per 0.50"] = nanper50
        df["Per 0.75"] = nanper75
        # df["Var"] = var
        df["Max"] = nanmax
        df["Prop Values"] = nanprop
        self.stats = df

        # Calculate proportion of sepsis positive
        sepsis = 0
        for label in self.labels:
            if max(label) == 1:
                sepsis += 1
        self.sepsis_prop = sepsis / len(self.labels)
        return self.stats, self.sepsis_prop

    def plot_histogram(self, covariate):
        if covariate not in self.column_names:
            return ValueError("{} not among covariates".format(covariate))
        index = self.column_names.index(covariate)
        x = self.data_matrix[:, index]
        plt.figure(0)
        plt.hist(x, bins=100)
        plt.xlabel(covariate)
        plt.show()
    
    def fill_NaN(self, form=0):
        for i, x in enumerate(self.data):
            df = pd.DataFrame(data=x)
            if form == 0:
                df.fillna(method='ffill', inplace=True)
                df.fillna(method='bfill', inplace=True)
                self.data[i] = df.to_numpy()
            else:
                self.data[i] = np.nan_to_num(self.data[i])
        self.data_matrix = np.concatenate(self.data, axis=0)  # concat data

    def add_feature(self, k):
        for i, x in enumerate(self.data):
            # add average of k previous data points
            length = x.shape[0]
            if k > 1:
                x_beg = np.expand_dims(x[0, :34], 0) * np.ones((k, 1))
                x_temp = x[:, :34]
                x_temp = np.concatenate((x_beg, x_temp), axis=0)
                x_conc = [np.mean(x_temp[j:j+1+k-1], axis=0) \
                    for j in range(length)]
                x_conc = np.stack(x_conc, axis=0)
                x = np.concatenate((x, x_conc), axis=1)
            # add handcrafted features
            a = (x[:, 39] > 58).astype(float)
            new_feat = np.expand_dims(a, 1)
            x = np.concatenate((x, new_feat), axis=1)
            self.data[i] = x
        self.data_matrix = np.concatenate(self.data, axis=0)  # concat data

    def plot_boxplot(self, show=False, savepath=None):
        data_box = []
        for i in range(self.num_features):
            x = self.data_matrix[:, i]
            x = x[~np.isnan(x)]  # remove nans
            if x.shape[0] > 0:
                x -= x.min()
                x /= x.max()
            data_box.append(x)
        fig, ax = plt.subplots()
        ax.boxplot(data_box, labels=self.column_names[:-1])
        plt.xticks(rotation=90)
        if show: plt.show()
        if savepath:
            fig.savefig(savepath)

    def plot_correlation(self, show=False, savepath=None):
        df = pd.DataFrame(data=self.data_matrix, \
            columns=self.column_names[:-1])
        corr_matrix = df.corr()
        fig, ax = plt.subplots()
        ax.matshow(corr_matrix)
        plt.title("correlation matrix")
        # ax.colorbar()
        if show: plt.show()
        if savepath:
            fig.savefig(savepath)

    def plot_fill_counts(self, show=False, savepath=None):
        nancount = np.sum(np.isnan(self.data_matrix).astype(int), 0)
        nanprop = 1 - (nancount).astype(float) / self.data_matrix.shape[0]

        ind = np.argsort(nanprop)
        nanprop = nanprop[ind]
        labels = [self.column_names[i] for i in ind]

        plt.figure(0, figsize=(12, 6))
        plt.bar(self.column_names[:-1], nanprop)
        plt.ylabel("proportion of filled values")
        plt.xticks(rotation=90)
        if show: plt.show()
        if savepath:
            fig.savefig(savepath)

        
    def plot_hospitalization_time(self, show=False, savepath=None):
        labels_sepsis = []
        max_time_s = 0
        for y in self.labels:
            if y.max() == 1:
                assert y[-1] == 1
                labels_sepsis.append(y)
                max_time_s = max(max_time_s, len(y))

        labels_nosepsis = []
        max_time_ns = 0
        for y in self.labels:
            if y.max() == 0:
                labels_nosepsis.append(y)
                max_time_ns = max(max_time_ns, len(y))

        # split data on time
        data_time_y = [ [] for i in range(max_time_s)]
        for i, y in enumerate(labels_sepsis):
            for j, yy in enumerate(y):
                data_time_y[j].append(yy)
        counts_s = [len(y) for y in data_time_y]

        # split data on time
        data_time_y = [ [] for i in range(max_time_ns)]
        for i, y in enumerate(labels_nosepsis):
            for j, yy in enumerate(y):
                data_time_y[j].append(yy)
        counts_ns = [len(y) for y in data_time_y]

        plt.subplot(1, 2, 1)
        plt.bar(np.arange(max_time_s), counts_s)
        plt.title("sepsis patients")
        plt.xlabel("hosp. time")
        plt.subplot(1, 2, 2)
        plt.bar(np.arange(max_time_ns), counts_ns)
        plt.title("no-sepsis patients")
        plt.xlabel("hosp. time")
        if show: plt.show()
        if savepath:
            fig.savefig(savepath)

    def plot_proportion_sepsis_sepsispatients(self, show=False, savepath=None):
        labels_sepsis = []
        max_time = 0
        for y in self.labels:
            if y.max() == 1:
                labels_sepsis.append(y)
                max_time = max(max_time, len(y))

        data_time_y = [ [] for i in range(max_time)]
        for i, y in enumerate(labels_sepsis):
            for j, yy in enumerate(y):
                data_time_y[j].append(yy)
        
        prop = [sum(y) / len(y) for y in data_time_y]
        plt.plot(prop)
        plt.xlabel("time")
        plt.ylabel("proportion sepsis label")
        if show: plt.show()
        if savepath:
            fig.savefig(savepath)


    def plot_proportion_sepsis(self, show=False, savepath=None):
        # split data on time
        max_time = max(self.lengths)
        data_time_y = [ [] for i in range(max_time)]
        for i, y in enumerate(self.labels):
            for j, yy in enumerate(y):
                data_time_y[j].append(yy)
        prop = [sum(y) / len(y) for y in data_time_y]
        plt.plot(prop)
        plt.xlabel("time")
        plt.ylabel("proportion sepsis label")
        if show: plt.show()
        if savepath:
            fig.savefig(savepath)

    def concatenate_features(self, k):
        data_cx, data_cy = [], []
        for data_x, data_y in zip(self.data, self.labels):
            # create new features
            length = data_x.shape[0]
            
            data_xn = [np.concatenate(data_x[i: i + k], axis=0) \
                for i in range(length - k)]
            data_xn = np.stack(data_xn, axis=0)
            data_cx.append(data_xn)
            data_yn = data_y[k:]
            data_cy.append(data_yn)
        data_matrix_c = np.concatenate(data_cx, axis=0)
        label_array_c = np.concatenate(data_cy, axis=0)
        return data_matrix_c, label_array_c
            

    

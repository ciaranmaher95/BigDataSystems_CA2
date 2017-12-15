import csv
import json
import collections
import random
import math
import pymongo
import numpy
import matplotlib.pyplot as plt
from numpy import dot
from pymongo import MongoClient
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

############################### INSERT DATA INTO MONGO DATABASE ###############################

client = MongoClient('localhost', 27017)
db = client['exams']
csvfile = open('C:\\Users\\Ciaran\\Desktop\\BigDataSystems\\CA2\\projectData.csv', 'r')

reader = csv.DictReader(csvfile, delimiter=';')

for row in reader:
	db.grades.insert_one(row)
	
############################### RETRIEVE DATA FROM THE DATABASE ###############################

data = []
studentData = []
finalGrade = []
	
for found in db.grades.find():
	data.append(int(found['Lab 1']))
	data.append(int(found['Christmas Test']))
	data.append(int(found['Lab 2']))
	data.append(int(found['Easter Test']))
	data.append(int(found['Lab3']))
	studentData.append(data)
	data = []
	finalGrade.append(int(found['Exam Grade']))

lab1 = []
lab2 = []
lab3 = []
christmasTest = []
easterTest = []
studentDataMean = []

for data in studentData:
	lab1.append(data[0])
	christmasTest.append(data[1])
	lab2.append(data[2])
	easterTest.append(data[3])
	lab3.append(data[4])

############################### RESCIPTIVE STATISTICS ###############################	

def mean(x):
	x = sum(x) / len(x)
	return round(x,2)
	
for i in range(0,len(studentData)):
	studentDataMean.append(mean(studentData[i]))
	
def median(x):
	length = len(x)
	midpoint = length // 2
	if length % 2 == 1:
		return no_of_texts_sorted[midpoint]
	else:
		lo = midpoint - 1
		hi = midpoint
		return (sorted(x)[lo] + sorted(x)[hi]) / 2

def rng(x):
	return max(x) - min(x)
	
def quantile(x, q):
	q_index = int(q * len(x))
	return sorted(x)[q_index]

def interquartileRange(x):
	return quantile(x, 0.75) - quantile(x, 0.25)
	
def de_mean(x):
	x_bar = mean(x)
	return [x_i - x_bar for x_i in x]

def	covariance(x,y):	
	n = len(x)	
	return numpy.dot(de_mean(x), de_mean(y)) / (n-1)
	
print("Mean of Lab 1: ", mean(lab1))
print("Mean of Christmas Resultss: ", mean(christmasTest))
print("Mean of Lab 2: ", mean(lab2))
print("Mean of Easter Results: ", mean(easterTest))
print("Mean of Lab 2: ", mean(lab3))
print("Mean of Final Exam Grade: ", mean(finalGrade))
print()
print("Median of Lab 1: ", median(lab1))
print("Median of Christmas Resultss: ", median(christmasTest))
print("Median of Lab 2: ", median(lab2))
print("Median of Easter Results: ", median(easterTest))
print("Median of Lab 2: ", median(lab3))
print("Median of Final Exam Grade: ", median(finalGrade))
print()
print("range of Lab 1: ", rng(lab1))
print("range of Christmas Resultss: ", rng(christmasTest))
print("range of Lab 2: ", rng(lab2))
print("range of Easter Results: ", rng(easterTest))
print("range of Lab 2: ", rng(lab3))
print("range of Final Exam Grade: ", rng(finalGrade))
print()
print("Interquartile Range of Lab 1: ", interquartileRange(lab1))
print("Interquartile Range of Christmas Resultss: ", interquartileRange(christmasTest))
print("Interquartile Range of Lab 2: ", interquartileRange(lab2))
print("Interquartile Range of Easter Results: ", interquartileRange(easterTest))
print("Interquartile Range of Lab 2: ", interquartileRange(lab3))
print("Interquartile Range of Final Exam Grade: ", interquartileRange(finalGrade))
print()
print('Covariance of Lab 1, Final Grade:',round(covariance(lab1,finalGrade),2))
print('Covariance of Christmas Test, Final Grade:',round(covariance(christmasTest,finalGrade),2))
print('Covariance of Lab 2, Final Grade:',round(covariance(lab2,finalGrade),2))
print('Covariance of Easter Grade, Final Grade:',round(covariance(easterTest,finalGrade),2))
print('Covariance of Lab 3, Final Grade:',round(covariance(lab3,finalGrade),2))

############################### 2D VISUALISATIONS ###############################

def scatterPlots(x,y,title,x_label,y_label):
	plt.scatter(x,y)
	plt.title(title)
	plt.xlabel(x_label, fontsize=12)
	plt.ylabel(y_label, fontsize=12)
	plt.show()

scatterPlots(lab1,finalGrade,'Lab 1 VS Final Grade (With Outliers)','Lab 1 Results','Final Exam Grade')
scatterPlots(christmasTest,finalGrade,'Christmas Test VS Final Grade (With Outliers)','Christmas Test Results','Final Exam Grade')
scatterPlots(lab2,finalGrade,'Lab 2 VS Final Grade (With Outliers)','Lab 2 Results','Final Exam Grade')
scatterPlots(easterTest,finalGrade,'Easter Test VS Final Grade (With Outliers)','Easter Test Results','Final Exam Grade')
scatterPlots(lab3,finalGrade,'Lab 3 VS Final Grade (With Outliers)','Lab 3 Results','Final Exam Grade')
scatterPlots(studentDataMean,finalGrade,'Mean of Student Data VS Final Grade','Mean of Student Data','Final Exam Grade')

############################### SIMPLE LINEAR REGRESSION ###############################

def regression(x,y,label):
	xbar = mean(x)
	ybar = mean(y)
	n = len(x)
	xy = []; 
	for i in range(0, n):
		xy.append(x[i]*y[i])
	num = (sum(xy)) - (n*xbar*ybar)
	xsqrdlist = []
	for i in range(0, n):
		xsqrd = x[i]**2
		xsqrdlist.append(xsqrd)
	den = sum(xsqrdlist) - (n*(xbar)**2)
	B1 = num / den
	B0 = ybar - B1*xbar
	residuals = []
	for i in range(0, n):
		residuals.append(round(B1*x[i]+B0,1))
	plt.scatter(x,y)
	X_plot = numpy.linspace(0,1,100)
	plt.plot(x,residuals,color='red')
	plt.xlabel(label, fontsize=12)
	plt.ylabel('Final Exam Grade', fontsize=12)
	plt.show()
	print('y=',round(B1,2),'x+',round(B0,2))

print()

print('Simple Linear Regression Models:')
regression(lab1,finalGrade,'Lab 1 Results')
regression(christmasTest,finalGrade,'Christmas Test Results')
regression(finalGrade,lab2,'Lab 2 Results')
regression(easterTest,finalGrade,'Easter Test Results')
regression(lab3,finalGrade,'Lab 3 Results')

print()
print('Lab 1, Final Grade Correlation Coefficient:')
print(numpy.corrcoef(lab1, finalGrade))
print()
print('Christmas Test, Final Grade Correlation Coefficient:')
print(numpy.corrcoef(christmasTest, finalGrade))
print()
print('Lab 2, Final Grade Correlation Coefficient:')
print(numpy.corrcoef(lab2, finalGrade))
print()
print('Easter Test, Final Grade Correlation Coefficient:')
print(numpy.corrcoef(easterTest, finalGrade))
print()
print('Lab 3, Final Grade Correlation Coefficient:')
print(numpy.corrcoef(lab3, finalGrade))

############################### MULTIPLE LINEAR REGRESSION ###############################

parttimejob = []

for found in db.grades.find():
	parttimejob.append(int(found['parttimejob']))

for i in range(0,len(studentData)):
	studentData[i].append(parttimejob[i])
	studentData[i].insert(0,1)

def predict_y(v,x):
	return dot(v,x)

def error(v,x,y):
	error = y-predict_y(v,x)
	return error
	
def squared_error(v,x,y):
	return error(v,x,y)**2 
	
def partial_difference_quotient(f,i,v,x,y,h):
	w = [v_j + (h if j == i else 0) for j, v_j in enumerate(v)]
	return (f(w,x,y)-f(v,x,y))/h

def stochastic_estimate_gradient(f,v,x,y,h):
	return [partial_difference_quotient(f,i,v,x,y,h) for i,_ in enumerate(v)]
	
def step(v,dir,stepsize):
	return [v_i - stepsize * dir_i for v_i, dir_i in zip(v,dir)]

def stochasticDescent(x,y):

	h = 0.001
	stepinit = 0.001
	
	v0 = [random.random() for x_i in x[0]]
	
	iterations_with_no_improvement = 0
	min_value = float("inf")
	min_v = None

	while iterations_with_no_improvement < 10:
		value = sum(squared_error(v0,x_i,y_i) for x_i, y_i in zip(x,y))
		if value < min_value: 
			iterations_with_no_improvement = 0
			min_v = v0
			min_value = value
			s0 = stepinit
		else:
			iterations_with_no_improvement += 1    
			s0 *= 0.9   
		indexes = numpy.random.permutation(len(x))
		for i in indexes:
			x_i = x[i]
			y_i = y[i]
			gradient_i = stochastic_estimate_gradient(squared_error,v0,x_i,y_i,h)
			v0 = step(v0,gradient_i,s0)
	return min_v

def variance(x):
    n = len(x)
    deviations = de_mean(x)
    return sum_of_squares(deviations) / (n - 1)
    
def standard_deviation(x):
    return math.sqrt(variance(x))
def sum_of_squares(x):
	return sum([x_i**2 for x_i in x])

def multiple_r_squared(v, x, y):
	sum_of_squared_errors = sum(error(v, x_i, y_i) ** 2 for x_i, y_i in zip(x, y))
	return 1.0 - sum_of_squared_errors / sum_of_squares(de_mean(y))
	
def normal_cdf(x, mu=0,sigma=1):
    return (1 + math.erf((x - mu) / math.sqrt(2) / sigma)) / 2 

def p_value(beta_hat_j, sigma_hat_j):
    if beta_hat_j > 0:
        return 2 * (1 - normal_cdf(beta_hat_j / sigma_hat_j))
    else:
        return 2 * normal_cdf(beta_hat_j / sigma_hat_j)  
		
def bootstrap_sample(data):
	list_data = list(data)
	rand_data = [random.choice(list_data) for _ in list_data]
	return rand_data

def bootstrap_statistic(x,y, stats_fn, num_samples):
	stats = []
	for i in range(num_samples):
		data = zip(x,y)
		sample_data = bootstrap_sample(data)
		x_sample,y_sample = zip(*sample_data)
		x = list(x_sample)
		y = list(y_sample)
		stat = stats_fn(x_sample,y_sample)
		stats.append(stat)
	return stats

estimate_v = stochasticDescent(studentData,finalGrade)
coefficients = bootstrap_statistic(studentData,finalGrade,stochasticDescent,100)
bootstrap_standard_errors = [standard_deviation([coefficient[i] for coefficient in coefficients]) for i in range(7)]
print()
print('mrs: ',multiple_r_squared(estimate_v,studentData,finalGrade))
for i in range(len(estimate_v)):
	print("i",i,"estimate_v",estimate_v[i],"error", bootstrap_standard_errors[i],"p-value", p_value(estimate_v[i], bootstrap_standard_errors[i]))

############################### PRINCIPLE COMPONENT ANALYSIS ###############################

for i in range(0,len(studentData)):
	del studentData[i][0]
	studentData[i].append(finalGrade[i])

#screeplot
pca =PCA()
plot_data = pca.fit_transform(studentData)
PC = pca.components_
PCEV=pca.explained_variance_
PCEVR=pca.explained_variance_ratio_
x=[i+1 for i in range(len(PC))]
plt.plot(x, PCEV)
plt.xlabel('Principal Component')
plt.ylabel('Variation Explained')
#plt.title('Scree-plot')
plt.show()

#principal components
pca = PCA(2)
plot_data = pca.fit_transform(studentData)
PC = pca.components_
PCEV=pca.explained_variance_
PCEVR=pca.explained_variance_ratio_
print("principal components:")
print(PC)
print("variance explained by each PC:")
print(PCEV)
print("proportion of variance explained by each PC:")
print(PCEVR)
print("transformed data:")
print(plot_data)
	
############################### CLUSTER ANALYSIS ###############################

#elbow plot
no_of_clusters = range(1,11)
avg_dist = []

for k in no_of_clusters:
	KMk = KMeans(k) 
	KMk.fit(studentData) 
	kmk_cluster_assignment = KMk.predict(studentData)
	avg_dist.append(sum(numpy.min(cdist(studentData, KMk.cluster_centers_,'euclidean'),axis=1)) / len(studentData))
plt.plot(no_of_clusters,avg_dist)
plt.xlabel('Number of Clusters')
plt.ylabel('Average Distance')
plt.show()

#cluster plot
KMr = KMeans(2) 
KMr.fit(studentData) 
plt.scatter(x=plot_data[:,0], y=plot_data[:,1], c=KMr.labels_,)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Scatterplot of Principal Components for 2 Clusters')
plt.show()

print()
print(KMr.cluster_centers_)
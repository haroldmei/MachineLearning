ordersPath = "/home/haroldmei/retail_db/orders/part-00000"
ordersFile = open(ordersPath)
ordersData = ordersFile.read()
orders = ordersData.splitlines()
for i in orders[:10]:
    print(i)

ordersMap = map(lambda o: (o.split(",")[0], o.split(",")[3]), orders)
for i in list(ordersMap)[:10]:
    print(i)

orderItemsPath = "/home/haroldmei/retail_db/order_items/part-00000"
orderItemsFile = open(orderItemsPath)
orderItemsData = orderItemsFile.read()
orderItems = orderItemsData.splitlines()
for i in orderItems[:10]:
    print(i)

orderItemsFiltered = filter(lambda oi: int(oi.split(",")[1]) == 2, orderItems)
orderItemsMap = map(lambda oi: float(oi.split(",")[4]), orderItemsFiltered)
#sum(orderItemsMap)
import functools as ft
ft.reduce(lambda x, y: x + y, orderItemsMap)


# pandas example
import pandas as pd
orderItemsPath = "/home/haroldmei/retail_db/order_items/part-00000"
orderItems = pd.read_csv(orderItemsPath, names=["order_item_id", "order_item_order_id", "order_item_product_id", "order_item_quantity", "order_item_subtotal", "order_item_product_price"])
orderItems[['order_item_id', 'order_item_subtotal']]
orderItems.query('order_item_order_id == 2')
orderItems.query('order_item_order_id == 2')['order_item_subtotal'].sum()
orderItems.groupby(['order_item_order_id'])['order_item_subtotal'].sum()

# Spark Modules
# Core – Transformations and Actions – APIs such as map, reduce, join, filter etc. They typically work on RDD
# Spark SQL and Data Frames – APIs and Spark SQL interface for batch processing on top of Data Frames or Data Sets (not available for Python)
# Structured Streaming – APIs and Spark SQL interface for stream data processing on top of Data Frames
# Machine Learning Pipelines – Machine Learning data pipelines to apply Machine Learning algorithms on top of Data Frames
# GraphX Pipelines

# Spark Data Structures
# We need to deal with 2 types of data structures in Spark – RDD and Data Frames.  We will see Data Structures in detail as part of the next topic.
# 
# RDD is there for quite some time and it is the low level data structure which spark uses to distribute the data between tasks while data is being processed
# RDD will be divided into partitions while data being processed. Each partition will be processed by one task.
# Data Frame is nothing but RDD with structure
# Typically we read data from file systems such as HDFS, S3, Azure Blob, Local file system etc
# Based on the file formats we need to use different APIs available in Spark to read data into RDD or Data Frame
# Spark uses HDFS APIs to read and/or write data from underlying file system

# Spark Framework
# Driver Program
# Spark Context
# Executors
# Executor Cache
# Executor Tasks
# Job
# Stage
# Task (Executor Tasks)

# Execution mode
# Local (for development)
# Standalone (for development)
# Mesos
# YARN


# RDD.mapPartition
# import itertools as it / it.chain.from_iterable(map(lambda s: s.split(" "), i))

# Python list is a heap of elements. We can manipulate Python list using APIs such as map, filter, reduce, set, sort, sorted etc.
# RDD stands for Resilient Distributed Dataset. It is a distributed list provided as part of Spark. Processed in a distributed fashion.
# We can manipulate Spark RDD using RDD functions such as map, flatMap, filter, reduceByKey etc.
# These APIs will not execute immediately, they only update the DAG and we need trigger execution by things like 'take(10)'.

# Characteristics of RDD
# In-memory
# Distributed
# Resilient

# Execution Life Cycle
# Data from files will be divided into RDD partitions and each partition is processed by a separate task
# By default, it will use HDFS block size (128 MB) to determine the partition size
# We can increase (cannot decrease) number of partitions by using an additional parameter in sc.textFile
# By default when data is loaded into memory each record will be serialized into Java object

# RDD Persistence
# Typically data will not be loaded into memory immediately when we create RDD as part of the program. It will be processed in real time by loading data into memory as it is processed. 
# If we have to retain RDD in memory for an extended period of time, then we have to use RDD Persistence.

# Let us see what happens when RDD is loaded into memory
# Serialize into Java Objects
# Get into memory
# As data is processed RDD partitions will be flushed out of memory as tasks are completed.
# We can persist the RDD partitions at different storage levels
# MEMORY_ONLY (default)
# MEMORY_AND_DISK
# DISK_ONLY
# and more

# Data Frames [Pandas? ]
# Many times data will have structure. Using RDD and then core APIs is some what tedious and cryptic. 
# Data Frames: 
# Flexible APIs (Data Frame native operations as well as SQL)
# Code will be readable
# Better organized and manageable
# Uses latest optimizers
# Process data in binary format
# Can generate execution plans based on statistics collected (for permanent tables such as Hive tables)
# We will talk about processing data using Data Frames in the next chapter. For now, we will be focusing on Core APIs

# Transformations (not execute)
# Row level transformations – map, flatMap, filter
# Joins – join, leftOuterJoin, rightOuterJoin
# Aggregations – reduceByKey, aggregateByKey
# Sorting data – sortByKey
# Group operations such as ranking – groupByKey
# Set operations – union, intersection

# Actions (execute)
# Previewing data – first, take, takeSample; converting RDD into the typical collection – collect
# Total aggregations – count, reduce; Total ranking – top; Saving files – saveAsTextFile, saveAsNewAPIHadoopFile

# Directed Acyclic Graph and Lazy Evaluation
# Thare are many APIs in Spark. But most of the APIs do not trigger execution of Spark job.
# When we create a Spark Context object it will procure resources from the cluster
# APIs used to read the data such as textFile as well as to process the data such as map, reduce, filter etc does not trigger immediate execution. They create variables of type RDD which also point to DAG.
# They run in driver program and build DAG. DAG will tell how it should execute. Each variable have a DAG associated with it.
# When APIs which are categorized as action (such as take, collect, saveAsTextFile) are used DAG associated with the variable is executed.
# In Scala, we can look at the DAG details by using toDebugString on top of the variables created.
# We can visualize DAG as part of Spark UI


#############################################################################
# Task 1: Get top N products by price in each category
# Convert the data to (k, v) using product category id as key and the entire product record as value
# Use groupByKey
# Use first and get first record and read the second element to regular python collection variable (productsPerCategory)
# Develop function to get top N products by price in that list
# Validate the function using productsPerCategory
# Invoke the function on output of groupByKey as part of flatMap
products = sc.textFile("/public/retail_db/products")
productsFiltered = products.filter(lambda p: p.split(",")[4] != "")
productsMap = productsFiltered.map(lambda p: (int(p.split(",")[1]), p))
productsGBCategory = productsMap.groupByKey()

def getTopNProducts(products, topN):
  return sorted(products, key=lambda k: float(k.split(",")[4]), reverse=True)[:topN]

topNProductsByCategory = productsGBCategory.flatMap(lambda p: getTopNProducts(list(p[1]), 3))
for i in topNProductsByCategory.take(10):
  print(i)

############################################################################
# Task 2: Get top N Priced products in each category
# Let us read products data into RDD
# Convert the data to (k, v) using product category id as key and the entire product record as value
# Use groupByKey
# Use first and get first record and read the second element to regular python collection variable (productsPerCategory)
# Develop function to get top N priced products in that list (simulating dense rank)
# Validate the function using productsPerCategory
# Invoke the function on output of groupByKey as part of flatMap

def getTopNPricedProducts(products, topN):
  import itertools as it
  productPrices = sorted(set(map(lambda p: float(p.split(",")[4]), products)), reverse=True)[:topN]
  productsSorted = sorted(products, key=lambda k: float(k.split(",")[4]), reverse=True)
  return it.takewhile(lambda product: float(product.split(",")[4]) in productPrices, productsSorted)

topNPricedProductsByCategory = productsGBCategory.flatMap(lambda p: getTopNPricedProducts(list(p[1]), 3))
for i in topNPricedProductsByCategory.take(10):
  print(i)

# data processing life cycle – row level transformations -> shuffling -> joins/aggregations

##########################################################################
# Sum Of Even Numbers
# parallelize: convert typical list to RDD
# collect: convert RDD to list
l = list(range(1, 100001))
lRDD = sc.parallelize(l) # convert to RDD;
lEven = lRDD.filter(lambda n: n % 2 == 0) # filter for even numbers;
sumEven = lEven.reduce(lambda x, y: x + y) # get sum of even numbers.

# Word count, per word statistics
lines = sc.textFile("/public/randomtextwriter/part-m-00000")
words = lines.flatMap(lambda line: line.split(" "))
wordTuples = words.map(lambda word: (word, 1))
wordCount = wordTuples.reduceByKey(lambda x, y: x + y)
wordCount.saveAsTextFile("/user/training/bootcamp/pyspark/wordcount")

# repartition or coalesce


#########################################################################
# groupByKey
orderItems = sc.textFile("/public/retail_db/order_items")
orderItemsMap = orderItems.map(lambda oi: (int(oi.split(",")[1]), float(oi.split(",")[4])))
orderItemsGBK = orderItemsMap.groupByKey(3)
orderItemsGBKMap = orderItemsGBK.map(lambda oi: (oi[0], sum(oi[1])))
for i in orderItemsGBKMap.take(10): print(i)

# reduceByKey
orderItems = sc.textFile("/public/retail_db/order_items")
orderItemsMap = orderItems.map(lambda oi: (int(oi.split(",")[1]), float(oi.split(",")[4])))
orderItemsRBK = orderItemsMap.reduceByKey(lambda x, y: x + y, 3)
for i in orderItemsRBK.take(10): print(i)

# aggregateByKey
orderItems = sc.textFile("/public/retail_db/order_items")
orderItemsMap = orderItems.map(lambda oi: (int(oi.split(",")[1]), float(oi.split(",")[4])))
orderItemsABK = orderItemsMap.aggregateByKey((0.0, 0), lambda x, y: (x[0] + y, x[1] + 1), lambda x, y: (x[0] + y[0], x[1] + y[1]), 3)
# orderItemsABK = orderItemsMap.aggregateByKey((0.0, 0), lambda x, y: (x[0] + y, x[1] + 1), lambda x, y: (x,y), 3)
# orderItemsABK = orderItemsMap.aggregateByKey((0.0, 0), lambda x, y: (x,y), lambda x, y: (x,y), 3)  # (32769, (((0.0, 0), 299.95), 299.95))  
for i in orderItemsABK.take(10): print(i)

# combiner


#########################################################################
# Get Daily Revenue (distinct, filter, map, join, reduceByKey)
# order_id, order_data, order_customer_id, order_status. 
orders = sc.textFile("/public/retail_db/orders") 
ordersFiltered = orders.filter(lambda o : o.split(",")[3] in ('COMPLETE', 'CLOSED')) #Only count 'COMPLETE' and 'CLOSED' orders.
ordersMap = ordersFiltered.map(lambda o : (int(o.split(",")[0]), o.split(",")[1]))
ordersByDate = ordersMap.groupByKey()

# order_item_id, order_item_order_id, order_item_product_id, order_item_quantity, order_item_subtotal, order_item_product_price
orderItems = sc.textFile("/public/retail_db/order_items") 
orderItemsMap = orderItems.map(lambda o : (int(o.split(",")[1]), float(o.split(",")[4])))

ordersJoin = ordersMap.join(orderItemsMap)
ordersJoinMap = ordersJoin.map(lambda o: o[1])

dailyRevenue = ordersJoinMap.reduceByKey(lambda x, y: x + y)
dailyRevenueSorted = dailyRevenue.sortByKey()
dailyRevenueSortedMap = dailyRevenueSorted.map(lambda oi: oi[0] + "," + str(oi[1]))
dailyRevenueSortedMap.saveAsTextFile("/user/haroldmei/bootcamp/pyspark/daily_revenue")


###########################################################################
# generate related variables
from numpy import mean
from numpy import std
from numpy.random import randn
from numpy.random import seed
from matplotlib import pyplot
# seed random number generator
seed(1)
# prepare data
data1 = 20 * randn(1000) + 100
data2 = data1 + (10 * randn(1000) + 50)
# summarize
print('data1: mean=%.3f stdv=%.3f' % (mean(data1), std(data1)))
print('data2: mean=%.3f stdv=%.3f' % (mean(data2), std(data2)))
# plot
pyplot.scatter(data1, data2)
pyplot.show()

##########################################################
import cgitb
cgitb.enable(display=0,logdir=".",format='')
print("Content-Type:text/html")
print()
print("<h1>hello world</h1>")
x=1/0

###########################################################
def gener(b):
  while b < 10:
    b += 1
    yield b
a = 0
gg = gener(a)
print(gg.__next__())


############################################################
def gen():
  a,b = 0,1
  print("START", end=" ")
  for i in range(4):
    yield b
    a,b = b,a+b
  print("END", end = " ")

def rungen():
  G = gen()
  while True:
    try:
      v = next(G)
      print(v, end=" ")
    except StopIteration:
      break

rungen() ##

print()
for v in gen(): ##
  print(v,end=" ")


###############################################################
# indefinite number of positional and keyword/value params
# func(1,2,3,a=1,b=2,c=3)
# (1, 2, 3)
# {'a': 1, 'b': 2, 'c': 3}
def func(*param, **kw):
  print(param)
  print(kw)


###############################################################
from string import Template 
s = Template('$who likes $what')
s = s.substitute(who='this', what='that')
print(s)

###############################################################
from multiprocessing import Process, Pipe
def f(conn):
  conn.send('This is sent trhough a pipe!')
  conn.close()

def g(conn):
  conn.send('This is sent trhough a pipe too!')
  conn.close()

if __name__ == '__main__':
  parent_conn, child1_conn = Pipe()
  #parent_conn, child2_conn = Pipe()
  p = Process(target = f, args = (child1_conn,))
  q = Process(target = g, args = (child1_conn,))
  q.start()
  p.start()
  print(parent_conn.recv())
  #print(parent_conn.recv())
  p.join()
  q.join()


################################################################
import os
from os.path import join, getsize
def f1(rootDir, exDir = 'CVS'):
  for root, dirs, files in os.walk(rootDir):
    print(root, 'consumes',)
    print(sum(getsize(join(root,name)) for name in files),)
    print('bytes in', len(files), 'files')
    if exDir in dirs:
      #print(dirs,exDir)
      dirs.remove(exDir)

f1('C:\\Users\\hmei\\Desktop\\DISK1','cab')


################################################################
def iter_idx_val(iterable):
  indices = range(len(iterable))
  for idx,val in zip(indices, iterable):
    yield idx,val

################################################################
from functools import reduce
X = lambda x: reduce(lambda y,z:y+z, range(1,x+1))
print(X(5))


################################################################
import warnings
def deprecated(func):
  def newFunc(*args, **kargs):
    warnings.warn('Call to deprecated function %s.' % (func.__name__), category=DeprecationWarning)
    return newFunc

@deprecated
def prod(x,y):
  return x*y

###############################################################.
import inspect
def f(x):
  """ this function does not do things"""
  pass
print(inspect.getdoc(f))



################################################################
#plot from samples
import numpy as np
from scipy.interpolate import UnivariateSpline
from matplotlib import pyplot as plt

N = 1000
n = N//10
s = np.random.normal(size=N)   # generate your data sample with N elements
#s = np.random.poisson(5, size=N)
p, x = np.histogram(s, bins=n) # bin it into n = N//10 bins
x = x[:-1] + (x[1] - x[0])/2   # convert bin edges to centers
f = UnivariateSpline(x, p, s=N)
plt.plot(x, f(x))
plt.show()


##########################################################
# convergence to point mass when n -> \infty
import numpy as np
from pylab import *
from scipy.stats import norm
def Fn(n):
  dx = 4.0/n
  X = np.arange(-2,2,dx)
  Y = norm.cdf(X * n ** 0.5)
  Y1 = [0 if x < 0 else 1 for x in X]
  plot(X, Y)
  plot(X, Y1)
  show()

#################################################################
# draw from poisson m samples and calculate the mean 
# repeat n times to form X_n;
# calculate the mean/variance of X_m, then plot 
def test_clt(m, n):
  s = np.zeros(n)
  for i in range(n):
    s[i] = np.mean(np.random.poisson(1.0, m)) - 1.0
    
  p, x = np.histogram(s, bins=n)
  x = x[:-1] + (x[1] - x[0])/2
  f = UnivariateSpline(x, p, s=n)
  plt.plot(x, f(x))
  plt.show()

#############################################################
import numpy as np
from matplotlib import pyplot as plt

# plot data with n bins;
def plotData(s):
  p, x = np.histogram(s, bins=100)
  x = x[:-1] + (x[1] - x[0])/2
  plt.plot(x, p)
  plt.show()

#############################################################
# relations of convergence
# quadratic mean => probability => distribution => probability to constant
# Slutzky's theorem

# minimum size of samples to achieve a certain requirement

# the standard deviation (aka variance) can be estimated by sample variance S_n.
# Berry-Esseen inequality, accuracy of the clt normal approximation
# Delta method 
# Almost Sure and L1 convergence (stronger than probability)
# The strong LLN
# Difference between weak/strong LLN
# asymptotically uniformly integrable
# 


###########################################################
import dbm

db=dbm.open('cache', 'c')

db['key1'] = b'value1'
db['key2'] = b'value2'
db['key3'] = b'value3'
db['key4'] = b'value4'
db['key5'] = b'value5'

db[b'key2'] = 'New value2'
db['key4'] = b'New value4'

print(db.get('key6', b'New value6'))
k = db.firstkey()
while k != None:
  print(k)
  k = db.nextkey(k)

db.close

##############################################################
import inspect
def callsig(function):
  """build a string with source code of the function call"""
  desc = inspect.getfullargspec(function)
  sign = ','.join(desc[0])
  if desc[1]:
    sign += ',*' + desc[1]
  if desc[2]:
    sign += ',**' + desc[2]
  if sign and sign[0] == ',':
    sign = sign[1:]
  return sign

def f(arg1, arg2 = None):
  pass

print(callsig(f))


############################################################
dict1 = {'key1':'val1','key2':'val2','key3':'val3'}
dict2 = {'key4':'val4','key5':'val5','key6':'val6'}
dict3 = dict1.update(dict2)
print(dict1)
print(dict2)
print(dict3)

############################################################
# generate function at run time
temp = """
def get_%s(data):
  print(data)
  return data[%s]
"""
def foo(name, idx):
  exec(temp % (name, idx), globals())

foo('a', 4)

#############################################################
# map, zip, reversed, sorted
wait_list = [1,2,3,4,5,6]
exec_list = [6,5,4,3,2,1]
#
comb_list = list(zip(wait_list, exec_list))
res = list(map(sorted, list(map(list, list(zip(*comb_list))))))
print(res[0][-1], res[1][-1])
#
comb_list = reversed(list(zip(wait_list, exec_list)))
res = list(map(sorted, list(map(list, list(zip(*comb_list))))))
print(res[0][-1], res[1][-1])
#
comb_list = list(zip(wait_list, exec_list))
res = list(map(max, list(zip(*comb_list))))
print(res[0], res[1])

##############################################################
class Foo(object):
  @classmethod
  def class_foo(cls):
    print("Class method for class %s" % cls)

Foo.class_foo()

# no decorator
class Foo(object):
  def class_foo(cls):
    print("Class method for class %s" % cls)
  class_foo = classmethod(class_foo)

Foo.class_foo()

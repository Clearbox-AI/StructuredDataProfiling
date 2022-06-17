
## StructuredDataProfiling

The StructuredDataProfiling is a Python library developed to assess structured datasets and to facilitate the creation of **data tests** through automated data profiling.

The library creates data tests in the form of **Expectations** using the [great_expectation](www.greatexpectations.io) framework. Expectations are 'declarative statements that a computer can evaluate and semantically meaningful to humans'. 

An expectation could be, for example, 'the sum of columns a and b should be equal to one' or 'the values in column c should be non-negative'.

StructuredDataProfiling runs a series of tests aimed at identifying statistics, rules, and constraints characterising a given dataset. The information generated by the profiler is collected by performing the following operations:

- Characterise uni- and bi-variate distributions.
- Identify data quality issues.
- Evaluate relationships between attributes (ex. column C is the difference between columns A and B) 
- Understand ontologies characterizing categorical data (column A contains names, while B contains geographical places).

For an overview of the library outputs please check the examples section.

# Installation
You can install StructuredDataProfiling by using the pip package manager:
`pip install -U structured-profiling
`
# Quickstart
You can import the profiler using
```python
from stucured_data_profiling import DataProfiler
```
You can import the profiler using
```python
profiler = DataProfiler('./csv_path.csv')
```
To start the profiling scripts run the method profile()
```python
profiler.profile()
```
The method info() outputs the results of the profiling process
```python
profiler.info()
```
# Profiling outputs
The profiler generates 3 json files describing the ingested dataset. These json files contain information about:
- Column profiles: it contains the statistical characterisation of the dataset columns. 
- Data quality: it highlights issues and limitations affecting the dataset.
- Data tests: it contains the data tests found by the profiler.

# Examples
You can find a couple of notebook examples in the [examples](./examples) folder.
# To-dos
- [ ] Support more data formats (Feather, Parquet)
- [ ] Add more Expectations
- [ ] Integrate PII identification using Presidio
- [ ] Compile part of the profiling routines using Cython 
- [ ] Write library tests
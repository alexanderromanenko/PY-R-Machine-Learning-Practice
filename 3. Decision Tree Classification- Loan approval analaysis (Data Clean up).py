import pandas as pd

# load data
df = pd.read_csv ("LoanStats3a.csv", sep = ',', skiprows = 1)

# select columns
df = df[['loan_amnt', 'term', 'sub_grade', 'emp_length', 'home_ownership', 'verification_status', 'purpose', 'loan_status']]

# drop all records with null values
df = df.dropna()

# drop all records with a loan status other than Fully Paid and Charged Off
df = df[df.loan_status.isin (['Fully Paid', 'Charged Off'])]

# drop all records where the employment length is not available
df = df[df.emp_length != 'n/a']

# convert the strings in term (duration of the loan) to numbers
df['term'] = df.term.apply (lambda x : int (x.split()[0]))

# convert the strings in emplen (employment length) to numbers
def empllengthprocess(x):
    x = x.split ('year')[0]
    if ('+') in x:
        return 12
    if ('<') in x:
        return 0
    else:
        return int (x)
df['emplen'] = df.emp_length.apply (lambda x : empllengthprocess (x))

# convert sub grade into a number
grades = ['G','F','E','D','C','B','A']
df['gradeencoding'] = df['sub_grade'].apply (lambda x : grades.index (x[0]) + (0.7 - 0.1 * float (x[1])))

# select columns
df = df[['loan_amnt', 'term', 'verification_status', 'gradeencoding', 'emplen', 'purpose', 'home_ownership', 'loan_status']]

# save to csv
df.to_csv ('Loans_processed.csv', index = False)
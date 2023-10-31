import csv

import pandas as pd
def format_results(data):
    d = {}
    df = pd.read_csv(data)
    for (columnName, columnData) in df.iteritems():
        column = list(columnData)
        for cell in column:
            cell = cell[1:-1]
            cell = cell.split(", ")
            capec = cell[0]
            cosine = cell[1]
            if capec not in d:
                d[capec] = [cosine]
            else:
                d[capec].append(cosine)
    return d

def format_capeccsv(data):
    result = {}
    reader = csv.reader(open(data, encoding="utf8"))
    next(reader)
    # Order of values in keys ['Name', 'Abstraction', 'Status', 'Description', 'Alternate Terms', 'Likelihood Of Attack',
    # 'Typical Severity', 'Related Attack Patterns', 'Execution Flow', 'Prerequisites', 'Skills Required', 'Resources
    # Required', 'Indicators', 'Consequences', 'Mitigations', 'Example Instances', 'Related Weaknesses',
    # 'Taxonomy Mappings', 'Notes', '']
    for row in reader:
        key = row[0]
        if key in result:
            # implement your duplicate row handling here
            pass
        result[key] = row[1:]
    return result
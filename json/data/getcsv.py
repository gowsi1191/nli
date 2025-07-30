# import os
# import json
# import xlsxwriter

# # Directory containing query files
# base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "")

# # Output Excel file
# output_path = os.path.join(base_dir, "queries_output.xlsx")
# workbook = xlsxwriter.Workbook(output_path)

# # Process query1.json to query5.json
# for i in range(1, 6):
#     file_path = os.path.join(base_dir, f"query{i}.json")
#     with open(file_path, "r") as f:
#         data = json.load(f)[0]  # Each file is a list with one dict

#     query = data["query"]
#     docs = data["documents"]

#     # Create a new sheet for each query
#     sheet = workbook.add_worksheet(f"Query{i}")
    
#     # Write query on top
#     sheet.write(0, 0, "Query:")
#     sheet.write(0, 1, query)
    
#     # Header row
#     sheet.write(2, 0, "Text")
#     sheet.write(2, 1, "Relevance")

#     # Write document data with relevance as "true"
#     for row_idx, doc in enumerate(docs, start=3):
#         sheet.write(row_idx, 0, doc["text"])
#         sheet.write(row_idx, 1, "true")

# workbook.close()
# print("✅ Excel file created with 5 sheets. Relevance is set to 'true' for all.")

import os
import json

# Directory containing query files
base_dir = os.path.dirname(os.path.abspath(__file__))

# Process query1.json to query5.json
for i in range(1, 32):
    file_path = os.path.join(base_dir, f"query{i}.json")
    with open(file_path, "r") as f:
        data = json.load(f)[0]  # Each file is a list with one dict

    query = data["query"]
    print(f"Query {i}: {query}")

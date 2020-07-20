import os
import re

n = 0

for filename in os.listdir(r'C:\Users\nathi_000\Desktop\Python Files\NLP Project\nlp-web-scrapping\src\experiment_configs'):
    m = re.search("exp\d{3}",filename)
    if m:
        if n < 2:
            print()
            print("--" * 50)
            print(filename)
            print("--" * 50)
            print()

            os.system("python run.py -cf .\experiment_configs\\" + filename)
            n += 1
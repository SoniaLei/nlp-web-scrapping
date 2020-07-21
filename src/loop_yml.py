import os
import re

# Needs to be run inside 'src' folder
path = os.path.abspath(os.path.dirname("")) + "\experiment_configs"

for filename in os.listdir(path)
    m = re.search("exp\d{3}",filename)
    if m:
        print()
        print("--" * 50)
        print(filename)
        print("--" * 50)
        print()

        os.system("python run.py -cf .\experiment_configs\\" + filename)
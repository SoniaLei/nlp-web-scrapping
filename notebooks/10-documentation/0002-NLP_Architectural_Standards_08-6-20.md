
# __NLP POD Architectural Standards__

## __Introduction__

The Natural Language Processing (NLP) Pod1 was stood up on the 20th of April 2020. 
The Epic goal for the Pod is to be able to Evaluate Tweets about financial markets to analyse the sentiment (Positive, Negative or Neutral) to ascertain if the market was in decline (Bearish)
or improving (Bullish).

After a Discovery Sprint and two sprints to improve data cleaning methods, create vectorisors (conversion of the data into a vector) and Sentiment Analysis Models (SAMs),
a Machine Learning Pipeline (MLP) is required.
This will enable "Modules" to be called and used in different combinations, until the optimum solution is discovered using [feature engineering](https://en.wikipedia.org/wiki/Feature_engineering)

The NLP Pod is using GitHub as the repository for the work created under [GNU General Public License v3.0](https://github.com/SoniaLei/nlp-web-scrapping/blob/development/LICENSE.md).

## __ML Pipleine Components and Modules__

The ML Pipeline is broken down into Components and Modules. Components are: Data cleaning/manipulation, ML Engine and Evaluation component. 
Modules are the parts that make up the components, such as Support Vector Machines, Bag of Words, TF-IDF, etc.

Using this modular approach enables the pipeline to call on (or off) certain modules to use when performing a pass of data through the pipeline.

### __MLP Component - Data Manipulation__

The Data Manipulation component consists of ways to clean the data and make it ready for o/p (output) to the ML Engine component.
Each part within the 

### __MLP Component - ML Engine__

This is the heart of the ML Pipeline and consists of Vectorisors and SAMs. 

### __MLP Component - Evaluation__

The evaluation component displays the results when data is passed through the pipeline. 

# __Standards and Procedures__

In order to maintain a haronised way of working and to enable creation of new and updated modules, standards are required to ensure that these components can fit seamlessly into the pipeline,
without disruption to other modules or components. All participants must adhere to the [Citizen Code of Conduct](https://github.com/SoniaLei/nlp-web-scrapping/blob/development/CODE_OF_CONDUCT.md).


## __Procedures for Modules__

To create a new module within a component, undertanding the inputs (i/p) and Outputs (o/p) is a key feature of modularisation. This enables plug and play feature within the pipleine.

- **All** new modules must conform to the [Class Diagram](insert link)

- Pull Requests, Sucessful Builds and Merge Pull Requests: Follow the instructions in the [ReadMe  file](https://github.com/SoniaLei/nlp-web-scrapping/blob/development/README.md)


### Addition of new components

New components must be accompanied with a task in the Current Sprint Backlog that has been approved by the Product Owner.

###Storage methods 

All files must be stored in  the [NLP Pod GitHUB Repository](https://github.com/SoniaLei/nlp-web-scrapping)

###Libraries and toolkits

Current libraries and toolkit versions that are required for modules are:

1. pytest
2. Calculator
3. pandas>=0.25.3,<0.26.0
4. numpy>=1.18.1,<1.19.0
5. scikit-learn>=0.22.1,<0.23.0
6. pytest>5.3.0
7. pytest-cov>2.8
8. beautifulsoup4>=4.8.0
9. nltk>3.0.0
10. spacy>2.1.0
11. wordcloud
12. holoviews
13. textblob

[from Requirements document](https://github.com/SoniaLei/nlp-web-scrapping/blob/development/requirements.txt)


###File Types

- Data cleaning outputs = .csv 
- Modules =  .py 
- Results = .md

###File Naming conventions  

So that files can be differentiated, and made unique, the follwing convention is to be used:
N.B.: do not use capital letters for filenames. 

*Four digit ID- Component-Module_type-details of the file . filetype*

This means moving away from and renaming modules as we move away from notebooks and towards a modular codebase.

Each four Digit ID must be unique - before creating a file, ensure you are using the next unused ID within the component ID range:

- 0000 to 0999:  These IDs are for Jupyter notebooks used in our initial work.

- 1000 to 1999: For **Data Component** modules.

- 2000 to 2999: For datasets created by the data Component Modules as .csv files.

- 3000 to 5999: For Vectorisor Modules within the **ML Engine Component**

- 6000 to 8999: For Sentiment Anlaysis Models within the **ML Engine Component**

- 9000 to 9999: For Evaluation code component modules.

Examples:

- 0001-rawdata-countVectorizer.ipynb

- 0003-basic_clean-countVectorizer.ipynb

- 1020-data-static_window.py

- 2050-data-dataset-method_ab.csv

- 2051-data-dataset-method_abcdef.csv

- 3020-vectorizer-tfidf.py

- 6020-sam-support_vector_machines.py

##Best Practice

To ensure the use of GitHub and coding Best Practice - 

1. Fetch at first daily logon, & before Push: this is to ensure that your local branch is aligned with the Dev and Prodcution Branch within GitHub before you commit changes.

2. Atomised Commits: Please ensure that each change is small enough to be validated and tested. It must have an appropriate commit message, indicating what changes have been made.
Please do not commit multiple changes to multiple files. This will enable bug tracing if faults are discovered and we need to roll back changes.

##Test Driven Development (TDD) methods and expectations.

*this requires input from the team*






# Kaiserreich-NLP
A natural language processing project about ideology in school books from the German Empire (1871-1918) using sentiment analysis, topic modelling and masked language modelling. This project was an exam project for a class in discourse analysis - so unlike my other repos, this project focuses more on the *interpretation* of machine learning models and natural language processing than building a solid pipeline.

For the project a sentiment analysis model was used to analyze 2000 school books from the German Empire. This model judges the sentiment (i.e. emotions) in sentences on a scale from 1 to 5. Using sentiment analysis for historical sources poses some very unique challenges and because of this the sentiment analysis model had to be used and interpreted in an untraditional way.  

The analysis folder contains an academic in-depth analysis of the output *in German*. Below is a technical description of the project *in English*. The code folder contains the code and parts of the output of the code.

The main focus of the project is the concept of *Mitteleuropa* - a historical term referring to Germany and its neighbouring countries. Other terms like *Deutschland* and *Europa* were analysed as part of the project as well. However, to really dive deep into the concept of *Mitteleuropa*, the sentiment analysis of this term was combined with topic modelling - a technique for determining what themes are commonly discussed in sentences containing *Mitteleuropa*.

## Technical overview
The school books are available at https://ddc.dwds.de/dstar/gei_digital/. If we want to research a term (e.g. *Mitteleuropa*), we download a .tsv file from the website containing all sentences with the term.

This .tsv file is then processed by read_tsv.py. This script does the following:

1) Data cleaning.

2) In each sentence, the term we want to analyze the usage of, is replaced by a semantically neutral token. I.e. "This is Mitteleuropa." becomes "This is 鬯.". Why? For sentiment analysis it is under normal circumstances incredibly important to use a model trained on text similar to the text we are analyzing. But there's no good model for 19th century German, and there are no annotated datasets we can use to train a model. Replacing the keyword we wish to research in this way is a solution to this issue - it prevents the vast majority of bias introduced by using a model trained on a vastly different dataset.

3) The sentiment (i.e. connotations, emotions) of each sentence is then analyzed using a BERT-model.

4) An overview of the sentiment analysis is saved to output_results, the full analysis is saved to datasets_with_sentiments, and plots showing changes to the sentiment during the period 1871-1918 is saved to output_stacked_area_chart.

The above process was used to research many different terms - e.g. *Afrika*, *Osteuropa* and *Klima*. After all of this, a topic modelling was carried out to really dig into what contexts *Mitteleuropa* is used in. The topic_modelling.py script does the following:

1) All sentences containing *Mitteleuropa* are divided into 3 sentiment categories using the output from read_tsv.py: positive, neutral and negative sentences. 

2) For each sentiment category, a number of topics are identified using Latent Dirichlet Allocation. Coincidentally, it was iteratively determined that each sentiment category could be subdivided into 3 major topics. 

3) An html file for each category is generated, containing useful information and pretty graphs about the topic modelling. The results in these html files were then analyzed by me. For example, it was clear that the positive sentences center on the following 3 topics: 1) Mitteleuropa as an industrial powerhouse led by Germany, 2) Mitteleuropa as a rich region with beautiful nature, and 3) Mitteleuropa as the center of civilization.

Additionally, a number of tools at https://ddc.dwds.de/dstar/gei_digital/ were used for analysing the data. This e.g. includes tools for analyzing which words frequently cooccur in the same sentences (collocations). 

I also experimented with finetuning the sentiment analysis model on smaller historical datasets - however, the results were disappointing due to the low quality and quantity of the available data. The approach with masked language modelling was a massive improvement over this.
# SOEN 471 - Big Data Analytics
## Team Members

| Name              | Student ID | GitHub        |
|-------------------|------------|---------------|
| Jeffrey Chan      | 40152579   | JeffreyCHChan |
| Joseph Mezzacappa | 40134799   | JosephMezza   |
| Rani Rafid        | 26975852   | Ra-Ni         |
| Arash Shafei      | 40156269   | Arash-Shafei  |


# Summary

Courses that an institution offers that align with the student’s interests and the institution’s requirements ultimately depend on the student’s capabilities to produce a plan that satisfies all requirements. 

The research question that this project hopes to address is “**What course should I take based on my interests?**” The end goal is to be able to suggest courses to users based on their preferences.

In this project we aim to provide suggestions by using the following three models: 1) K-Means clustering tinkering with various hyperparameters (initial mean points, number of classes, etc.); 2) Content-Based Recommendations [1]; 3) Clustering via Gaussian Mixture Model (GMM). The K-means model will be used as a baseline for performance while a content-based model and GMM will aim to beat the performance set by the K-Means.

The dataset includes features such as course code composed of subject and catalog number, course name/title, degree/career, and credit/class unit attribute, as well as course description and course requisites. This is provided to us by Concordia’s Open Data through an API call as well as through raw data presented in a Comma Separated Value (CSV) file [2] and will be updated as time progresses by the school as courses become discontinued and courses are created.

User and Item profiles must contain the two following criteria as they will be used as inputs to any models: **Code/Program**; **Interests/Description** the interests may be split between categories of terms and definitions. Preprocessing of the course description to extract the keywords will be done using DBpedia Spotlight through HTTP requests to perform automated tagging. scikit-learn will remove stop words and perform term frequency-inverse document frequency(tf-idf) to decrease the weight of common words ,and increase the weight on words specific for the course

Using these two items (**Code/Program** and **Interests/Description**) we can create profiles for items and users to use as inputs for the three models presented before.

Evaluation metrics that will be experimented will include those found on the sci-kit learn website, and the silhouette coefficient [3] for clustering as well as others mentioned in references [3] and [4].

Problems we anticipate include feedback, as currently there is no way to obtain implicit or explicit feedback. Another is the lack of a truth value to the recommendations made by the model.

The recommendation system uses a model based on the Resource Description Framework (RDF) as it presents information in a flexible and well structured manner. At its core, RDF consists of a subject-predicate-object triple, which can be modified at design-time and run-time. Additionally, RDF resources are identified by their unique address, the URI. This presents a distinct advantage compared to other modeling techniques, such as UML, which are designed to be used at design-time. Indeed, RDF facilitates the use of machine learning applications.

[1] https://towardsdatascience.com/build-recommendation-system-with-pyspark-using-alternating-least-squares-als-matrix-factorisation-ebe1ad2e7679

[2] https://www.concordia.ca/web/open-data.html

[3] https://www.analyticsvidhya.com/blog/2020/10/quick-guide-to-evaluation-metrics-for-supervised-and-unsupervised-machine-learning/

[4] https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
 

# Python Version
    Python 3.10.9

# Install necessary libraries:

    pip install -r requirement.txt 

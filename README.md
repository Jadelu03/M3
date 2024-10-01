# Module 3: PLS, Modeling and Classification
We will learn to plot Partial Least Squares (PLS) results, model using Partial Least Squares Regression (PLSR) and do classification analyses.
## Assignment 1: PLS and Modeling
PLS as an initial analysis can provide insight into your data and relations
between the explanatory dataset X, the response dataset y and the
observations. Practice using two datasets involving 40 wine samples- the
C dataset has 17 chemical measurements on the wine samples and the X
dataset is NMR data on the same samples. Can NMR replace the chemical
measures? Do any of the datasets classify wine color? Use the data
M3_Wine_Chem.csv and M3_Wine_NMR.csv.
- Preprocess the X and C datasheets.
- Make a PLS plot using X as the explanatory variables and C as the response variables. Color observations by wine type.
- Make Effects Plots to identify specific NMR signals that significantly associate with chemical measures. Present the results in a table and show one example of an Effects
plot. Do these results correspond to the PLS figure?
- Carry out PLSR using PLSRegression, creating a plot of variance explained versus components used. Plot as R2 for both training and test data (using train_test_split).
- Make a prediction plot using y and predicted y. What is the R2 value? How many PCs are predicted to be optimal for the model? Is it a good model?
- Extra analyses might include comparing the PLS plot to a PCA, reorganizing the categories to improve results (hint- it involves wine type) and selecting a subset of X and C variables that improve the model. For the regression analysis, you can include MSE and Q2 calculations in your modeling efforts or try another modeling type, such as PCR.
## Assignment 1 continued: Classification
PLS plots and regression modeling focus on the variables measured and only indirectly involve the treatments of the study (in this case wine type). Here we focus on the ability of the data to classify the different treatments (wine color). Keep in mind that classification involves categorizing the levels of a treatment based on the response variables, not studying the response variables themselves.
- Use KNeighborsClassifier to classify wine type by first the chemical measures, and then
the NMR variables. Do the chemical measures or NMR classify wine type better? Show
the confusion charts. Discuss the results in terms of accuracy and precision. Why might
classification of wine samples be important?
- Extra analyses include using another model (ie LogisticRegression), limiting the variables
to those found influential in Effects Plots, creating decision boundary plots and
reorganizing the levels to improve classification.
## Assignment 2
Be sure to examine and follow the rubric for this assignment. The report should contain an
introduction, results, discussion and conclusion. Answers should be more comprehensive than
above. Take care to format figures, tables etc. Justify your actions and conclusions.
Can fecal microbiome sampling replace biochemical biomarkers for identifying different
stages of bowel cancer? 45 bowel cancer patients (from stage 1 to 4, with 4 being worst) were
tested using 9 blood biomarkers and fecal microbiome sampling containing 23 bacteria (from
Order to Species). Metadata includes patient's declared gender. Carry out preprocessing, PLS,
modeling and classification on the microbiome and biomarker data, along with any relevant
analyses learned in M1 + M2.
Consider:
- Which biomarkers and bacteria are associated with the different stages of bowel
cancer?
- Can you identify the relevant variables using Effects Plots?
- How well does the microbiome model the biomarkers?
- What can the bacteria identify that biomarkers cannot?
- Do bacteria or biomarkers classify the disease levels better?
- Are there any issues with the design?


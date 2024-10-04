import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load your dataset
# Drop rows with NaN values

ydf = pd.read_csv('E:\Year_2_Semester_1\chemometrics\Exercises\M3\M3_project\C_Mod.csv')  # Adjust this to your data format
xdf = pd.read_csv('E:\Year_2_Semester_1\chemometrics\Exercises\M3\M3_project\X_Mod.csv')

xdrops = xdf[xdf.isna().any(axis=1)].index.tolist()
cdrops = ydf[ydf.isna().any(axis=1)].index.tolist()
nans = xdrops + cdrops
print("NaN rows dropped:", nans)

# Re-assign cdf and xdf
ydf = ydf.drop(nans)
xdf = xdf.drop(nans)

y = ydf
X = xdf


le = LabelEncoder()
y['color'] = le.fit_transform(y['color'])  # Encode 'white' and 'red' to numbers
print(f'color is {y['color']}')

# Chemical measures
X_chemical = y[['Malic', 'Ethanol', 'Total', 'Volatile', 'Lactic_Acid', 'Tartaric', 
                   'Glucose', 'Density', 'Folin', 'Glycerol', 'Gluconic', 'Sorbic', 
                   'CO2', 'Citric', 'Methanol', 'Ethylacetate', 'pH']]
y = y['color']

# NMR features (assuming they are in columns after chemical measures)
X_nmr = X.drop(columns=['Sample', 'color'])  # Drop 'Sample' and target variable


from sklearn.model_selection import train_test_split

# Split for chemical features
X_train_chem, X_test_chem, y_train, y_test = train_test_split(X_chemical, y, test_size=0.3, random_state=42)

# Split for NMR features
X_train_nmr, X_test_nmr, _, _ = train_test_split(X_nmr, y, test_size=0.3, random_state=42)


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize classifier
knn_chem = KNeighborsClassifier()

# Fit the model
knn_chem.fit(X_train_chem, y_train)

# Predictions
y_pred_chem = knn_chem.predict(X_test_chem)

# Evaluation
print("Chemical Measures Classification Report:\n", classification_report(y_test, y_pred_chem))
cm_chem = confusion_matrix(y_test, y_pred_chem)
sns.heatmap(cm_chem, annot=True, fmt='d')
plt.title('Confusion Matrix - Chemical Measures')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# Initialize classifier
knn_nmr = KNeighborsClassifier()

# Fit the model
knn_nmr.fit(X_train_nmr, y_train)

# Predictions
y_pred_nmr = knn_nmr.predict(X_test_nmr)

# Evaluation
print("NMR Features Classification Report:\n", classification_report(y_test, y_pred_nmr))
cm_nmr = confusion_matrix(y_test, y_pred_nmr)
sns.heatmap(cm_nmr, annot=True, fmt='d')
plt.title('Confusion Matrix - NMR Features')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

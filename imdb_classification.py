import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('ggplot')
sns.set_palette('Set2')

# Load the data
X_train = pd.read_csv("IMDB dataset/X_train.csv")
y_train = pd.read_csv("IMDB dataset/y_train.csv", header=None)
X_test = pd.read_csv("IMDB dataset/X_test.csv")
y_test = pd.read_csv("IMDB dataset/y_test.csv", header=None)

# Make sure X_train and y_train have the same number of samples
print(f"X_train shape before alignment: {X_train.shape}")
print(f"y_train shape before alignment: {y_train.shape}")

# Align the lengths
if len(X_train) != len(y_train):
    # Use the minimum length
    min_len = min(len(X_train), len(y_train))
    X_train = X_train.iloc[:min_len]
    y_train = y_train.iloc[:min_len]

# Do the same for test data
if len(X_test) != len(y_test):
    min_len = min(len(X_test), len(y_test))
    X_test = X_test.iloc[:min_len]
    y_test = y_test.iloc[:min_len]

# Convert to numpy arrays immediately to avoid feature name issues
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

print(f"X_train shape after alignment: {X_train.shape}")
print(f"X_test shape after alignment: {X_test.shape}")

# Define normalizers
normalizers = {
    'None': None,
    'StandardScaler': StandardScaler(),
    'MinMaxScaler': MinMaxScaler(),
    'MaxAbsScaler': MaxAbsScaler()
}

# Define 4-fold cross-validation
kf = KFold(n_splits=4, shuffle=True, random_state=42)

# Results dictionary to store metrics
results = {}

# Function to normalize data (no need to convert from DataFrame to array anymore)
def normalize_data(X_train, X_test, normalizer):
    if normalizer is None:
        return X_train, X_test
    X_train_norm = normalizer.fit_transform(X_train)
    X_test_norm = normalizer.transform(X_test)
    return X_train_norm, X_test_norm

# Function to visualize cross-validation results
def plot_cv_results(model_name, cv_results, param_name, scoring='accuracy'):
    plt.figure(figsize=(10, 6))
    
    for normalizer_name in cv_results.keys():
        params = cv_results[normalizer_name]['params']
        scores = cv_results[normalizer_name]['mean_test_score']
        std = cv_results[normalizer_name]['std_test_score']
        
        plt.plot(params, scores, 'o-', label=normalizer_name)
        plt.fill_between(params, scores - std, scores + std, alpha=0.1)
    
    plt.xlabel(param_name)
    plt.ylabel(f'Cross-validated {scoring}')
    plt.title(f'{model_name} - Parameter Tuning with Different Normalizations')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{model_name}_cv_results.png')
    plt.close()

# Function to plot confusion matrix
def plot_confusion_matrix(cm, model_name, normalizer_name):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - {model_name} ({normalizer_name})')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f'{model_name}_{normalizer_name}_confusion_matrix.png')
    plt.close()

# Function to plot feature importance
def plot_feature_importance(importance, title, filename, top_n=20):
    # Get indices of top features
    indices = np.argsort(importance)[-top_n:]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(indices)), importance[indices], align='center')
    plt.yticks(range(len(indices)), [f'Feature {i}' for i in indices])
    plt.xlabel('Importance')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_training_vs_test_accuracy(results):
    """Plot training vs test accuracy for all models and normalizers"""
    plt.figure(figsize=(15, 10))
    
    # Prepare data for bar chart
    models = []
    train_accs = []
    test_accs = []
    colors = []
    normalizers = []
    
    model_colors = {
        'KNN': 'royalblue',
        'MultinomialNB': 'forestgreen',
        'RandomForest': 'firebrick',
        'GradientBoosting': 'darkorange'
    }
    
    for model_name, model_results in results.items():
        for norm_name, metrics in model_results.items():
            models.append(f"{model_name}\n({norm_name})")
            train_accs.append(metrics['train_accuracy'])
            test_accs.append(metrics['test_accuracy'])
            colors.append(model_colors[model_name])
            normalizers.append(norm_name)
    
    x = np.arange(len(models))
    width = 0.35
    
    plt.bar(x - width/2, train_accs, width, label='Training Accuracy', color=colors, alpha=0.7)
    plt.bar(x + width/2, test_accs, width, label='Test Accuracy', color=colors, alpha=1.0)
    
    plt.xlabel('Model & Normalization')
    plt.ylabel('Accuracy')
    plt.title('Training vs Test Accuracy for Different Models and Normalizations')
    plt.xticks(x, models, rotation=45, ha='right')
    plt.ylim(0.5, 1.0)
    
    # Add value labels
    for i, v in enumerate(train_accs):
        plt.text(i - width/2, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    
    for i, v in enumerate(test_accs):
        plt.text(i + width/2, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_vs_test_accuracy.png')
    plt.close()

def plot_training_times(results):
    """Plot training times for all models and normalizers"""
    plt.figure(figsize=(15, 8))
    
    models = []
    times = []
    colors = []
    
    model_colors = {
        'KNN': 'royalblue',
        'MultinomialNB': 'forestgreen',
        'RandomForest': 'firebrick',
        'GradientBoosting': 'darkorange'
    }
    
    for model_name, model_results in results.items():
        for norm_name, metrics in model_results.items():
            models.append(f"{model_name}\n({norm_name})")
            times.append(metrics['training_time'])
            colors.append(model_colors[model_name])
    
    plt.bar(models, times, color=colors)
    plt.xlabel('Model & Normalization')
    plt.ylabel('Training Time (seconds)')
    plt.title('Training Time for Different Models and Normalizations')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('training_times.png')
    plt.close()

#========== KNN Classifier ==========
print("\n========== KNN Classifier ==========")
knn_results = {}

# Parameters to tune
k_values = np.arange(3, 16, 2)
distance_metrics = ['euclidean', 'manhattan']

for norm_name, normalizer in normalizers.items():
    print(f"\nNormalizer: {norm_name}")
    start_time = time.time()
    
    # Normalize data
    X_train_norm, X_test_norm = normalize_data(X_train, X_test, normalizer)
    
    best_k = None
    best_metric = None
    best_score = 0
    cv_results = {'params': [], 'mean_test_score': [], 'std_test_score': []}
    
    for distance in distance_metrics:
        for k in k_values:
            knn = KNeighborsClassifier(n_neighbors=k, metric=distance, n_jobs=-1)
            scores = cross_val_score(knn, X_train_norm, y_train, cv=kf, scoring='accuracy')
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            
            param_name = f"k={k}, metric={distance}"
            cv_results['params'].append(param_name)
            cv_results['mean_test_score'].append(mean_score)
            cv_results['std_test_score'].append(std_score)
            
            print(f"k={k}, metric={distance}, CV accuracy: {mean_score:.4f} ± {std_score:.4f}")
            
            if mean_score > best_score:
                best_score = mean_score
                best_k = k
                best_metric = distance
    
    # Train with best parameters
    best_knn = KNeighborsClassifier(n_neighbors=best_k, metric=best_metric, n_jobs=-1)
    best_knn.fit(X_train_norm, y_train)
    
    # Predict on test data
    y_pred = best_knn.predict(X_test_norm)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    training_time = time.time() - start_time
    
    print(f"\nBest parameters: k={best_k}, metric={best_metric}")
    print(f"Training accuracy (CV): {best_score:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Training time: {training_time:.2f} seconds")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, 'KNN', norm_name)
    
    # Store results
    knn_results[norm_name] = {
        'cv_results': cv_results,
        'best_params': {'k': best_k, 'metric': best_metric},
        'train_accuracy': best_score,
        'test_accuracy': test_accuracy,
        'training_time': training_time,
        'classifier': best_knn
    }

# Visualize KNN CV results
for norm_name, results_dict in knn_results.items():
    plt.figure(figsize=(12, 8))
    cv_results = results_dict['cv_results']
    
    plt.bar(cv_results['params'], cv_results['mean_test_score'], yerr=cv_results['std_test_score'],
            alpha=0.7, capsize=5)
    
    plt.xlabel('Parameters (k, metric)')
    plt.ylabel('Cross-validated accuracy')
    plt.title(f'KNN - Parameter Tuning with {norm_name} Normalization')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f'KNN_{norm_name}_cv_results.png')
    plt.close()

# Record KNN results
results['KNN'] = knn_results

#========== Multinomial Naive Bayes ==========
print("\n========== Multinomial Naive Bayes ==========")
nb_results = {}

# Multinomial NB requires non-negative features
# For StandardScaler, we'll clip negative values to 0
for norm_name, normalizer in normalizers.items():
    print(f"\nNormalizer: {norm_name}")
    start_time = time.time()
    
    # Normalize data
    X_train_norm, X_test_norm = normalize_data(X_train, X_test, normalizer)
    
    # For StandardScaler, clip negative values
    if norm_name == 'StandardScaler':
        X_train_norm = np.clip(X_train_norm, 0, None)
        X_test_norm = np.clip(X_test_norm, 0, None)
    
    # Parameters to tune
    alpha_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    best_alpha = None
    best_score = 0
    cv_results = {'params': alpha_values, 'mean_test_score': [], 'std_test_score': []}
    
    for alpha in alpha_values:
        mnb = MultinomialNB(alpha=alpha)
        scores = cross_val_score(mnb, X_train_norm, y_train, cv=kf, scoring='accuracy')
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        cv_results['mean_test_score'].append(mean_score)
        cv_results['std_test_score'].append(std_score)
        
        print(f"alpha={alpha}, CV accuracy: {mean_score:.4f} ± {std_score:.4f}")
        
        if mean_score > best_score:
            best_score = mean_score
            best_alpha = alpha
    
    # Train with best parameters
    best_mnb = MultinomialNB(alpha=best_alpha)
    best_mnb.fit(X_train_norm, y_train)
    
    # Predict on test data
    y_pred = best_mnb.predict(X_test_norm)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    training_time = time.time() - start_time
    
    print(f"\nBest parameter: alpha={best_alpha}")
    print(f"Training accuracy (CV): {best_score:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Training time: {training_time:.2f} seconds")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, 'MultinomialNB', norm_name)
    
    # Feature importance (coefficient magnitude)
    feature_importance = np.abs(best_mnb.feature_log_prob_[1] - best_mnb.feature_log_prob_[0])
    plot_feature_importance(feature_importance, 
                          f'Multinomial NB Feature Importance ({norm_name})',
                          f'MultinomialNB_{norm_name}_feature_importance.png')
    
    # Store results
    nb_results[norm_name] = {
        'cv_results': cv_results,
        'best_params': {'alpha': best_alpha},
        'train_accuracy': best_score,
        'test_accuracy': test_accuracy,
        'training_time': training_time,
        'classifier': best_mnb,
        'feature_importance': feature_importance
    }

# Plot Naive Bayes CV results
for norm_name, results_dict in nb_results.items():
    plot_cv_results('MultinomialNB', {norm_name: results_dict['cv_results']}, 'Alpha')

# Record Naive Bayes results
results['MultinomialNB'] = nb_results

#========== Random Forest ==========
print("\n========== Random Forest ==========")
rf_results = {}

for norm_name, normalizer in normalizers.items():
    print(f"\nNormalizer: {norm_name}")
    start_time = time.time()
    
    # Normalize data
    X_train_norm, X_test_norm = normalize_data(X_train, X_test, normalizer)
    
    # Parameters to tune
    n_estimators_values = [50, 100, 200]
    max_depth_values = [None, 10, 20, 30]
    
    best_n_estimators = None
    best_max_depth = None
    best_score = 0
    cv_results = {'params': [], 'mean_test_score': [], 'std_test_score': []}
    
    for n_est in n_estimators_values:
        for max_d in max_depth_values:
            rf = RandomForestClassifier(n_estimators=n_est, max_depth=max_d, 
                                       random_state=42, n_jobs=-1)
            scores = cross_val_score(rf, X_train_norm, y_train, cv=kf, scoring='accuracy')
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            
            param_name = f"n_est={n_est}, max_depth={max_d}"
            cv_results['params'].append(param_name)
            cv_results['mean_test_score'].append(mean_score)
            cv_results['std_test_score'].append(std_score)
            
            print(f"n_estimators={n_est}, max_depth={max_d}, CV accuracy: {mean_score:.4f} ± {std_score:.4f}")
            
            if mean_score > best_score:
                best_score = mean_score
                best_n_estimators = n_est
                best_max_depth = max_d
    
    # Train with best parameters
    best_rf = RandomForestClassifier(n_estimators=best_n_estimators, max_depth=best_max_depth, 
                                   random_state=42, n_jobs=-1)
    best_rf.fit(X_train_norm, y_train)
    
    # Predict on test data
    y_pred = best_rf.predict(X_test_norm)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    training_time = time.time() - start_time
    
    print(f"\nBest parameters: n_estimators={best_n_estimators}, max_depth={best_max_depth}")
    print(f"Training accuracy (CV): {best_score:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Training time: {training_time:.2f} seconds")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, 'RandomForest', norm_name)
    
    # Feature importance
    plot_feature_importance(best_rf.feature_importances_, 
                          f'Random Forest Feature Importance ({norm_name})',
                          f'RandomForest_{norm_name}_feature_importance.png')
    
    # Store results
    rf_results[norm_name] = {
        'cv_results': cv_results,
        'best_params': {'n_estimators': best_n_estimators, 'max_depth': best_max_depth},
        'train_accuracy': best_score,
        'test_accuracy': test_accuracy,
        'training_time': training_time,
        'classifier': best_rf,
        'feature_importance': best_rf.feature_importances_
    }

# Visualize RF CV results
for norm_name, results_dict in rf_results.items():
    plt.figure(figsize=(15, 8))
    cv_results = results_dict['cv_results']
    
    plt.bar(cv_results['params'], cv_results['mean_test_score'], yerr=cv_results['std_test_score'],
            alpha=0.7, capsize=5)
    
    plt.xlabel('Parameters (n_estimators, max_depth)')
    plt.ylabel('Cross-validated accuracy')
    plt.title(f'Random Forest - Parameter Tuning with {norm_name} Normalization')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f'RandomForest_{norm_name}_cv_results.png')
    plt.close()

# Record Random Forest results
results['RandomForest'] = rf_results

#========== Gradient Boosting ==========
print("\n========== Gradient Boosting ==========")
gb_results = {}

for norm_name, normalizer in normalizers.items():
    print(f"\nNormalizer: {norm_name}")
    start_time = time.time()
    
    # Normalize data
    X_train_norm, X_test_norm = normalize_data(X_train, X_test, normalizer)
    
    # Parameters to tune
    n_estimators_values = [50, 100, 200]
    learning_rate_values = [0.05, 0.1, 0.2]
    max_depth_values = [3, 5, 7]
    
    best_n_estimators = None
    best_learning_rate = None
    best_max_depth = None
    best_score = 0
    cv_results = {'params': [], 'mean_test_score': [], 'std_test_score': []}
    
    for n_est in n_estimators_values:
        for lr in learning_rate_values:
            for max_d in max_depth_values:
                gb = GradientBoostingClassifier(n_estimators=n_est, learning_rate=lr, 
                                              max_depth=max_d, random_state=42)
                scores = cross_val_score(gb, X_train_norm, y_train, cv=kf, scoring='accuracy')
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                
                param_name = f"n_est={n_est}, lr={lr}, max_depth={max_d}"
                cv_results['params'].append(param_name)
                cv_results['mean_test_score'].append(mean_score)
                cv_results['std_test_score'].append(std_score)
                
                print(f"n_estimators={n_est}, learning_rate={lr}, max_depth={max_d}, CV accuracy: {mean_score:.4f} ± {std_score:.4f}")
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_n_estimators = n_est
                    best_learning_rate = lr
                    best_max_depth = max_d
    
    # Train with best parameters
    best_gb = GradientBoostingClassifier(n_estimators=best_n_estimators, learning_rate=best_learning_rate, 
                                       max_depth=best_max_depth, random_state=42)
    best_gb.fit(X_train_norm, y_train)
    
    # Predict on test data
    y_pred = best_gb.predict(X_test_norm)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    training_time = time.time() - start_time
    
    print(f"\nBest parameters: n_estimators={best_n_estimators}, learning_rate={best_learning_rate}, max_depth={best_max_depth}")
    print(f"Training accuracy (CV): {best_score:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Training time: {training_time:.2f} seconds")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, 'GradientBoosting', norm_name)
    
    # Feature importance
    plot_feature_importance(best_gb.feature_importances_, 
                          f'Gradient Boosting Feature Importance ({norm_name})',
                          f'GradientBoosting_{norm_name}_feature_importance.png')
    
    # Store results
    gb_results[norm_name] = {
        'cv_results': cv_results,
        'best_params': {'n_estimators': best_n_estimators, 'learning_rate': best_learning_rate, 'max_depth': best_max_depth},
        'train_accuracy': best_score,
        'test_accuracy': test_accuracy,
        'training_time': training_time,
        'classifier': best_gb,
        'feature_importance': best_gb.feature_importances_
    }

# Visualize the best configurations for Gradient Boosting CV
for norm_name, results_dict in gb_results.items():
    # Create a subset of top 10 configurations for visualization
    cv_results = results_dict['cv_results']
    indices = np.argsort(cv_results['mean_test_score'])[-10:]
    
    plt.figure(figsize=(15, 8))
    plt.bar([cv_results['params'][i] for i in indices], 
            [cv_results['mean_test_score'][i] for i in indices],
            yerr=[cv_results['std_test_score'][i] for i in indices],
            alpha=0.7, capsize=5)
    
    plt.xlabel('Parameters (n_estimators, learning_rate, max_depth)')
    plt.ylabel('Cross-validated accuracy')
    plt.title(f'Gradient Boosting - Top 10 Parameter Configs with {norm_name} Normalization')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f'GradientBoosting_{norm_name}_cv_results.png')
    plt.close()

# Record Gradient Boosting results
results['GradientBoosting'] = gb_results

#========== Compare Classifiers ==========
print("\n========== Comparing Classifiers ==========")

# Plot all results
plot_training_vs_test_accuracy(results)
plot_training_times(results)

# Find the best overall model based on test accuracy
best_model = None
best_normalizer = None
best_accuracy = 0
best_model_set = None

for model_name, model_results in results.items():
    for norm_name, metrics in model_results.items():
        if metrics['test_accuracy'] > best_accuracy:
            best_accuracy = metrics['test_accuracy']
            best_model = model_name
            best_normalizer = norm_name
            best_model_set = metrics

print(f"\nBest model: {best_model} with {best_normalizer} normalization")
print(f"Test accuracy: {best_accuracy:.4f}")
print(f"Training time: {best_model_set['training_time']:.2f} seconds")
print(f"Best parameters: {best_model_set['best_params']}")

# Analyze misclassified samples for the best model
print("\n========== Misclassified Samples Analysis ==========")

# Get the best classifier
best_classifier = best_model_set['classifier']
X_norm, _ = normalize_data(X_test, X_test, normalizers[best_normalizer])
y_pred = best_classifier.predict(X_norm)

# Find misclassified samples
misclassified = np.where(y_pred != y_test)[0]
print(f"Number of misclassified samples: {len(misclassified)}")

# Analyze misclassified samples
if len(misclassified) > 0:
    # First few misclassified samples
    sample_indices = misclassified[:5] if len(misclassified) >= 5 else misclassified
    
    for idx in sample_indices:
        actual = y_test[idx]
        predicted = y_pred[idx]
        print(f"\nSample {idx}")
        print(f"Actual: {'Positive' if actual == 1 else 'Negative'}")
        print(f"Predicted: {'Positive' if predicted == 1 else 'Negative'}")
        
        # For models that provide feature importance, analyze what caused the misclassification
        if hasattr(best_classifier, 'feature_importances_') or best_model == 'MultinomialNB':
            feature_importance = (best_model_set['feature_importance'] 
                                 if 'feature_importance' in best_model_set 
                                 else np.zeros(X_test.shape[1]))
            
            # Get the highest-valued features in this sample
            sample_features = X_test[idx]  # X_test is now a numpy array
            feature_indices = np.argsort(-sample_features)[:5]  # Top 5 features with highest values
            
            print("Top features in this sample:")
            for i, feature_idx in enumerate(feature_indices):
                value = sample_features[feature_idx]
                importance = feature_importance[feature_idx] if len(feature_importance) > 0 else 0
                print(f"  Feature {feature_idx}: value={value:.4f}, importance={importance:.4f}")

# Create a summary plot of test accuracy for all models
plt.figure(figsize=(12, 8))

models = []
accuracies = []
colors = []

model_colors = {
    'KNN': 'royalblue',
    'MultinomialNB': 'forestgreen',
    'RandomForest': 'firebrick',
    'GradientBoosting': 'darkorange'
}

for model_name, model_results in results.items():
    for norm_name, metrics in model_results.items():
        models.append(f"{model_name}\n({norm_name})")
        accuracies.append(metrics['test_accuracy'])
        colors.append(model_colors[model_name])

# Sort by accuracy
sort_idx = np.argsort(accuracies)[::-1]
models = [models[i] for i in sort_idx]
accuracies = [accuracies[i] for i in sort_idx]
colors = [colors[i] for i in sort_idx]

# Plot
plt.figure(figsize=(15, 8))
bars = plt.barh(models, accuracies, color=colors)

# Add value labels
for i, bar in enumerate(bars):
    plt.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2, 
            f'{accuracies[i]:.4f}', va='center')

plt.xlabel('Test Accuracy')
plt.title('Model Performance Comparison (Test Accuracy)')
plt.xlim(0.5, 1.0)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('model_comparison_summary.png')
plt.close()

print("\nAnalysis complete. All plots have been saved.") 
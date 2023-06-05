# Machine Learning Test Project: Diagnosis Prediction Based on Symptoms and Risk Factors

This is a test project in order to experiment with machine learning algorithms.

## Usage
Download the data from [Infermedica](https://infermedica.com/test-cases) and save it as `test_cases.json` in the directory [data](data/).
Run [main.py](diagnosis_prediction/main.py). The results will be saved to the directory [results](results/).
Download the data from [Infermedica](https://infermedica.com/test-cases) and save it as `test_cases.json` in the directory [data](data/).
Run [main.py](diagnosis_prediction/main.py). The results will be saved to the directory [results](results/).

## Project Description
### Problem Definition
The goal is to predict diagnoses of patients based on features like sex, age, symptoms and risk factors.

The data is a list of JSON objects reflecting the different patients. The JSON objects are nested and have the following structure:
 
```python
patient = {
    "public_test_case_id": 6425,
    "public_test_case_name": "21-year-old female with an umblilical erythema",
    "public_test_case_source": "100 Cases in Dermatology. Morris-Jones R, Powell AM, Benton E. CRC Press; 2011, page 19-22",
    "expected_condition_name": "Systemic lupus erythematosus",
    "expected_condition_common_name": "Undescended testicle",
    "expected_condition_id": "c_1133",
    "expected_condition_position_range": "top_3",
    "expected_condition_icd10_codes":["T78.2"],
    "evidence":{
        "present":[
            {"id": "s_156", "name": "Headache, recent", "type": "symptom", "initial": true},
            {"id": "s_310", "name": "Dry mouth", "type": "risk_factor", "initial": false}],
        "absent":[{"id": "s_101","name": "Itching after a hot shower or bath","type": "symtom","initial": false}],
        "unknown":[]
        },
    "test_case_passed": true,
    "predicted_conditions":[
        {"id": "c_1116","name": "Anaphylaxis","common_name": "Foreign body aspiration","probability": 0.8197},
        {"id": "c_77","name": "Seborrheic dermatitis","common_name": "Paronychia","probability": 0.174}
        ],
    "api_payload":{
        "sex": "female",
        "age":{"value": 11,"unit": "year"},
        "evidence":[
            {"id": "s_2307", "choice_id": "present"},
            {"id": "s_510", "choice_id": "absent", "source": "initial"}
            ]
        }
    }
```
Note that this is a randomized data entry.

The field `api_payload` repeats some information: `patient["api_payload"]["evidence"]` reflects the information of `patient["evidence"]`. `patient["api_payload"["evidence"]["choice_id"]` can take the values "present", "absent" or "unknown"; and `patient["api_payload"["evidence"]["source"]` can take the value "initial".

The fields `expected_condition_name`, `expected_condition_common_name`, `expected_condition_id`, `expected_condition_position_range`, `expected_condition_icd10_codes` are related to the target of the prediction.
The fields `predicted_conditions`, `test_case_passed` are the result of the prediction model from Infermedica and are ignored here.
The other fields can be used as input for the prediction: `public_test_case_id`, `public_test_case_name`, `public_test_case_source`, `evidence`, `api_payload`.

### Data Exploration & Data Loading
The data exploration is done in the jupyter notebook [data_exploration.ipynb](diagnosis_prediction/data_exploration.jpynb) and its resulting files are saved to the directory [data_exploration](data_exploration/). Some more detailed findings are stated in the jupyter notebook. Here, the most important insights are summarized.

The given data is not tabular. In order to load the data into a (tabular) pandas dataframe, some manual manipulation is necessary to extract the desired columns. 

The data contains 373 rows. The possible target columns `expected_condition_id`, `expected_condition_name`, `expected_condition_icd10_codes` take over 260 distinct values, that is over 70% of the number of rows. In order to make sensible predictions, the target needs to be grouped. For this, the ICD-10 code given by the column `expected_condition_icd10_codes` are categorized by means of ICD-10 chapter definitions which are stored in [icd10_chapters_definition.csv](data/icd10_chapters_definition.csv). The resulting column is called `expected_condition_icd10_blocks`.

Since more than one ICD-10 code can be given in the `expected_condition_icd10_codes` column, there can also be multiple corresponding ICD-10 chapters in the target. To avoid a multilabel classification problem, the target is mapped to unique identifiers.

With this mapping, the target `expected_condition_icd10_blocks` takes 32 distinct values (8.6% of the number of rows) which will be easier to predict. Although, there are still 15 different values that occur less than 5 times.

The age information is given as `value` and `unit` which can be either `"year"` or `"month"`. For an easier learning, the age value is converted to a consistent unit `"month"`.

All of this is performed in the [DataLoader](diagnosis_prediction/data_loader.py) class.  The result is a feature matrix with the following columns
- `sex: str`
- `age_in_months: int`
- `evidence_present: List[str]`
- `evidence_absent: List[str]`

and a target vector:
- `expected_condition_icd10_blocks: str`.

The resulting data is clean: It has no missing cells, no duplicated rows and does not seem to have incorrect information. Therefore, no data cleaning is required.

Note that we dropped the information `public_test_case_source` and whether an evidence has been initial since they seem less important and would blow up the feature matrix.

### Data Preprocessing
The `sex` is a binary categorical variable since it takes the 2 values `"male"` and `"female"`. It will be therefore encoded with an `OrdinalEncoder`.

The `age_in_months` is a numerical variable which will be scaled with the `StandardScaler` since scaling is important for machine learning algorithms that calculate distances are that assume normality.

`evidence_present` and `evidence_absent` hold lists of strings of different lengths. To encode these, we create a column for each possible string (evidence) and set its value to 
- 1 if that evidence is present,
- -1 if that evidence is absent,
- 0 if that evidence is not given or unknown. 

This is done in the [EvidenceEncoder](diagnosis_prediction/evidence_encoder.py) class and yields 750 columns when applied to the whole dataset. Note that this exceeds the number of rows which makes a model prone to overfitting.

Because of that, different dimensionality reduction algorithms are applied before the training to reduce the number of features.
Another option is to neglect the absent evidence. But this still results in 706 columns when encoding the evidence of the whole dataset.

We optionally add features that count the number of present and absent evidence for each row. This could be especially helpful for tree based models. Those counts are scaled with the `MinMaxScaler` to be in the range [-1,1].

The [PipelineBuilder](diagnosis_prediction/pipeline_builder.py) class is putting all steps together to build the final pipeline which is ready to be trained.

### Model Training
In order to find a good predictor, different pipelines are tested by varying the following things:

- whether to include the evidence counts
- whether to include the absent evidence,
- dimensionality reduction algorithm and the resulting number of dimensions,
- classifier.

This is implemented in the [PerformanceEvaluator](diagnosis_prediction/performance_evaluator.py) class with the `train_and_evaluate` method.

As a baseline model, `sklearn`'s `Dummyclassifier` is used predicting the most frequent target. Apart from that, the following machine learning algorithms are tested:

- K-nearest neighbors algorithm,
- logistic regression,
- perceptron,
- multi-layer perceptron,
- decision tree,
- random forest,
- extra trees,
- gradient boosting,
- support vector machine (with linear and radial basis function kernel),
- linear discriminant analysis,
- passive aggressive algorithm.


### Model Evaluation
Because of the small size of the dataset, cross validation is used to get a more reliable performance evaluation. For this, we choose a 5 fold splitting resulting in validation sets containing approximately 75 samples.

As performance metric, the accuracy score is used.

We find that the baseline model has a test accuracy of around 16.1% whereas the best models go up to 56.5% (Multi-layer perceptron and linear support vector machine). The train accuracy on the other hand is nearly 100% which clearly shows overfitting.

### Model Tuning
The overfitting is attempted to be reduced by tuning hyperparameters that control the model complexity:

- number of neighbors for k-nearest neighbors algorithm,
- (inverse) regularization strength for logistic regression, support vector machine, passive aggressive algorithm, perceptron, multi-layer perceptron,
- number of neurons for multi-layer perceptron,
- shrinkage for linear discriminant analysis,
- maximum tree depth for decision tree, extra trees, random forest, gradient boosting,
- number of estimators for extra trees, random forest, gradient boosting.

This is done via a grid search with the `perform_gridsearch` method of the [PerformanceEvaluator](diagnosis_prediction/performance_evaluator.py) class.
The resulting best pipelines for all classifiers are stored in [best_parameters.csv](results/best_parameters.csv). There are also plots depicting the train and test accuracy when varying those hyperparameters.

Unfortunately, the improvement of the test accuracy from the model tunding is less than 3% for every classifier.

The best predictor is the following:

- classifier: linear support vector machine
- dimensionality reduction: principal component analysis with 250 dimensions
- absent evidence is not included
- number of evidence are not count
- inverse regularization strength `C` of 0.1

It can be obtained via the `get_best_predictor` method of the [PerformanceEvaluator](diagnosis_prediction/performance_evaluator.py) class and yields a test accuracy of approximately 57.3% and a train accuracy of 100%.

### Fit Quality
For a better judgement of the fit quality, the learning curves of the best predictors are plotted. (Note that the number of samples in the train set cannot be smaller than the number of components of the PCA).

We find converged curves for the train and test accuracy whith a large gap between them. This indicates that the train set is too small. 

### Conclusion & Outlook
The baseline model yields a test accuracy of 16.1% whereas the best predictor, a linear support vector machine, achieves a test accuracy of 57.3% which is more than 3 times better.

However, the best predictor still overfits severely which is predominantly caused by a lack of data. Therefore, the main focus for imporovement should be to get more data.

Apart from that, it could be sensible to try out other feature reduction methods. One could analyze the feature importance or group the evidence into categories. 
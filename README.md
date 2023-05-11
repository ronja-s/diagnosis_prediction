# Machine Learning Test Project: Diagnosis Prediction Based on Symptoms and Risk Factors

## Usage
Download the data from https://infermedica.com/test-cases and save it as test_cases.json in the directory ./data.
Run main.py.

## Project Description
### Problem Definition
The goal is to predict conditions (diagnoses) of patients based on features like sex, age, symptoms and risk factors.

The data test_cases.json is a list of JSON objects reflecting the different patients. The JSON objects are nested and have the following structure:

```json
patient = {"public_test_case_id":int,"public_test_case_name":str,"public_test_case_source":str,"expected_condition_name":str,"expected_condition_common_name":str,"expected_condition_id":str,"expected_condition_position_range":str,"expected_condition_icd10_codes":[str],"evidence":{"present":[{"id":str,"name":str,"type":str,"initial":bool}],"absent":[{"id":str,"name":str,"type":str,"initial":bool}],"unknown":[{"id":str,"name":str,"type":str,"initial":bool}]},"test_case_passed":bool,"predicted_conditions":[{"id":str,"name":str,"common_name":str,"probability":float}],"api_payload":{"sex":str,"age":{"value":int,"unit":str},"evidence":[{"id":str,"choice_id":str},{"id":str,"choice_id":str,"source":str}]}}
```

Note that the field `api_payload` repeats some information: `patient["api_payload"]["evidence"]` reflects the information of `patient["evidence"]`. `patient["api_payload"["evidence"]["choice_id"]` can take the values "present", "absent" or "unknown"; and `patient["api_payload"["evidence"]["source"]` can take the value "initial".

The fields `expected_condition_name`, `expected_condition_common_name`, `expected_condition_id`, `expected_condition_position_range`, `expected_condition_icd10_codes` are related to the target of the prediction.
The fields `predicted_conditions`, `test_case_passed` are the result of the prediction model from Infermedica and are ignored here.
The other fields can be used as input for the prediction: `public_test_case_id`, `public_test_case_name`, `public_test_case_source`, `evidence`, `api_payload`.

### Data Loading
The given data is not tabular. In order to load the data into a (tabular) pandas dataframe, some manual manipulation was performed.
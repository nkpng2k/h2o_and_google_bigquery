from h2o_and_google_bigquery import GoogleH2OIntegration
import h2oai_client
from h2oai_client import Client, ModelParameters


# Intialize object of class GoogleH2OIntegration
# Query data from Google BigQuery
dataset = 'iris_dataset'
pred_table = 'pred_table'
bq_auth = '/Users/npng/Downloads/h2o-project-090347f40536.json'
g_h2o = GoogleH2OIntegration(dataset, pred_table, bq_auth=bq_auth)
train_df = g_h2o.bigquery_query()
test_df = g_h2o.bigquery_query()

fp_train_csv = 'Users/npng/Downloads/train.csv'
fp_test_csv = 'Users/npng/Downloads/test.csv'

train_df.to_csv(fp_train_csv)
test_df.to_csv(fp_test_csv)

# Initialize H2O Driverless AI
address = "http://127.0.0.1:12345"
username = 'npng'
password = 'npng'

h2oai = Client(address=address, username=username, password=password)

train = h2oai.create_dataset_sync(fp_train_csv)
test = h2oai.create_dataset_sync(fp_train_csv)

# set the parameters you want to pass to the UI
target = "Survived"
drop_cols = ['PassengerId']
weight_col = None
fold_col = None
time_col = '[AUTO]'

# Pre-set parameters to pass model
is_classification = True
enable_gpus = True
seed=True
scorer_str = 'auc'

# Pre-sent accuracy knobs
accuracy_value = 10
time_value = 7
interpretability = 5

experiment = h2oai.start_experiment_sync(ModelParameters(
    # Datasets
    dataset_key=train.key,
    testset_key=test.key,
    validset_key='', # No validatoin dataset provided

    # Columns
    target_col=target,
    cols_to_drop=drop_cols,
    weight_col=weight_col,
    fold_col=fold_col,
    time_col=time_col,

    # Parameters
    is_classification=is_classification,
    enable_gpus=enable_gpus,
    seed=seed,
    accuracy=accuracy_value,
    time=time_value,
    interpretability=interpretability,
    scorer=scorer_str
))

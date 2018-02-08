from h2o_and_google_bigquery import GoogleH2OIntegration
import h2oai_client
from h2oai_client import Client, ModelParameters


# Intialize object of class GoogleH2OIntegration
# Query data from Google BigQuery
dataset = 'iris_dataset'
pred_table = 'pred_table'
bq_auth = '/Users/npng/Downloads/h2o-project-090347f40536.json'
g_h2o = GoogleH2OIntegration(dataset, pred_table, bq_auth=bq_auth)
df = g_h2o.bigquery_query()

y = np.zeros((df.shape[0], 1))

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2)
df = df.drop(['Id','Species'], axis=1)

# Initialize H2O Driverless AI
address = "http://127.0.0.1:12345"
username = 'npng'
password = 'npng'

h2oai = Client(address=address, username=username, password=password)

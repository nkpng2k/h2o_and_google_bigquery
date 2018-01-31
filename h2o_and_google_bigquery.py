from google.cloud import bigquery
from h2o.automl import H2OAutoML
from sklearn.model_selection import train_test_split
import numpy as np
import h2o


class GoogleH2OIntegration(object):

    def __init__(self, dataset, pred_table_name):
        """
        INPUT: dataset (STRING) - name of dataset from Google bigquery
               pred_table_name (STRING) - name preferred for
        """
        self.client = bigquery.Client()
        self.dataset = self.client.dataset(dataset)
        self.col_name = ['test_id', 'prediction']
        self.col_type = ['INTEGER', 'STRING']
        self.mode = ['required', 'required']
        self.SCHEMA = []
        self._set_schema(self.col_name,
                         self.col_type,
                         self.mode)
        self.pred_table_ref = self.dataset.table(pred_table_name)
        self.pred_table = bigquery.Table(self.pred_table_ref,
                                         schema=self.SCHEMA)
        self.pred_table = self.client.create_table(self.pred_table)

    def bigquery_query(self):
        query = self._multiline()
        query_job = self.client.query(query)
        df = query_job.to_dataframe()
        return df

    def _multiline(self):
        print "Enter/Paste your content. Ctrl-D to save it."
        contents = []
        while True:
            try:
                line = raw_input("")
            except EOFError:
                break
            contents.append(line)
        return " ".join(contents)

    def _set_schema(self, col_name, col_type, mode):
        for i in xrange(len(col_name)):
            one_col = bigquery.SchemaField(col_name[i],
                                           col_type[i],
                                           mode=mode[i])
            self.SCHEMA.append(one_col)


if __name__ == "__main__":
    g_h2o = GoogleH2OIntegration('iris_dataset', 'prediction_table')
    df = g_h2o.bigquery_query()

    # Note: Do some feature engineering
    y = np.zeros((df.shape[0], 1))

    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2)
    df = df.drop('species', axis=1)

    h2o.init(nthreads=-1, max_mem_size='2g', ip="127.0.0.1", port=54321)
    train_col = list(X_train.columns)
    test_col = list(X_test.columns)
    train = h2o.H2OFrame.from_python(X_train, column_names=train_col)
    test = h2o.H2OFrame.from_python(X_test, column_names=test_col)

    x = train.columns
    y = 'species'
    x.remove(y)

    aml = H2OAutoML(max_runtime_secs=30)
    aml.train(x=x, y=y, training_frame=train, leaderboard_frame=test)
    lb = aml.leaderboard
    print lb

    all_data = h2o.H2OFrame.from_python(df, column_names=x)
    predictions = aml.leader.predict(all_data).as_data_frame()

    add_to_table = zip(xrange(predictions.shape[0]), predictions['predict'])

    g_h2o.client.insert_rows(g_h2o.pred_table, add_to_table)

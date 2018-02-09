from google.cloud import bigquery
from h2o.automl import H2OAutoML
from sklearn.model_selection import train_test_split
import h2oai_client
from h2oai_client import Client, ModelParameters
import numpy as np
import pandas as pd
import h2o


class GoogleH2OIntegration(object):

    def __init__(self, dataset, pred_table_name, bq_auth=None):
        """
        Constructs class object of GoogleH2OIntegration.
        NOTE: if argument pred_table_name points to an already existing table
              in BigQuery, that table will be deleted
        INPUT: dataset (STRING) - name of dataset from Google bigquery
               pred_table_name (STRING) - name for new table in bigquery
        ATTRIBUTES: self.client - Initialized bigquery client using API,
                                  assumes google cloud authentications handled
                                  already via google cloud sdk
                    self.dataset - dataset reference for bigquery API
                    self.col_name/self.col_type/self.mode - lists of strings
                                  necessary for creating 2 column prediction
                                  table schema
                    self.SCHEMA - self._set_schema() is called becomes a list
                                  of type SchemaField
                    self.pred_table_ref - table reference for bigquery API
                    self.pred_table - instance of new bigquery table
        """
        if bq_auth.split('.')[-1] == 'json':
            self.client = bigquery.Client.from_service_account_json(bq_auth)
        else:
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
        try:
            self.client.get_table(self.pred_table_ref)
            self.client.delete_table(self.pred_table_ref)
            self.pred_table = self.client.create_table(self.pred_table)
            print ('There was an existing table')
        except:
            self.pred_table = self.client.create_table(self.pred_table)
            print ('There was NO existing table')

    def bigquery_query(self):
        """
        METHOD: prompts user to enter StandardSQL query as INPUT

        NOTE: Enter first line of query, hit return,
              next line of query, return.
              Once finished press CTRL + D to complete query

        OUTPUT: dataframe of results from bigquery query
        """
        query = self._multiline()
        query_job = self.client.query(query)
        df = query_job.to_dataframe()
        return df

    def write_to_table(self, test_ids, predictions):
        """
        Takes test_ids and predictions and adds them to new predictions table
        INPUT: test_ids (LIST of INTEGERS) - foreign keys for SQL JOIN
               predictions (LIST of STRINGS) - current schema set by default
                                               for strings, can be changed.
                                               All predictions from test set
        """
        to_table = zip(test_ids, predictions)
        self.client.insert_rows(self.pred_table, to_table)
        print ("Success")

    def h2o_automl(self, X_train, X_test, target, drop_cols, h2o_args,
                   aml_args):
        """
        Initializes an instance of H2O and runs H2O AutoML to identify and
        return top model
        INPUT: X_train (DATAFRAME) - training data with target column
               X_test (DATAFRAME) - validation data (can have target column)
               target (STRING) - name of target column
               drop_cols (LIST of STRINGS) - list of all columns to be ignored
                                             in training
               h2o_args (DICT of kwargs) - dictionary containing all desired
                                           arguments for initializing H2O
               aml_args (DICT of kwargs) - dictionary containing all desired
                                           arguments for initializing AutoML
        OUTPUT: aml - trained AutoML object containing best model (aml.leader)
        """
        h2o.init(**h2o_args)

        train_col = list(X_train.columns)
        test_col = list(X_test.columns)

        train = h2o.H2OFrame.from_python(X_train, column_names=train_col)
        test = h2o.H2OFrame.from_python(X_test, column_names=test_col)

        x = train.columns
        for col in remove_cols:
            x.remove(col)

        aml = H2OAutoML(**aml_args)
        aml.train(x=x, y=target, training_frame=train, leaderboard_frame=test)
        lb = aml.leaderboard
        print (lb)

        return aml

    def h2o_dia(self, client_args, model_params, train_df, test_df, fp_local,
                fp_dai, valid_df=None):
        """
        INPUT: client_args (DICT) - dictionary containing kwargs for launching
                                    h2oai_client.client
               model_params (DICT) - dictionary containing kwargs that will be
                                     passed to h2oai_client.ModelParameters
               train_df (DATAFRAME) - training dataframe resulting from
                                      bigquery_query()
               test_df (DATAFRAME) - test dataframe resulting from
                                     bigquery_query()
               fp_local (LIST of STRINGS) - strings are filepaths to local
                                            directory where H2O DAI will look.
                                            ex: [train_fp, test_fp]
               fp_dai (LIST of STRINGS) - strings are filepaths formatted to
                                          where H2O DAI checks for data on
                                          server
                                          ex: [train_fp, test_fp]
               valid_df (OPTIONAL DATAFRAME) - dataframe for validation
                                               dataset. NOTE: if NOT None, both
                                               fp_local and fp_dai will need
                                               to be length of 3 instead of 2
                                               ex: [train_fp, test_fp, valid_fp]
        OUTPUT: test_preds (DATAFRAME) - pandas dataframe containing
                                         predictions made on test dataset
        """
        train_df.to_csv(fp_local[0])
        test_df.to_csv(fp_local[1])

        h2o_ai = Client(**client_args)

        train = h2o_ai.create_dataset_sync(fp_dai[0])
        test = h2o_ai.create_dataset_sync(fp_dai[1])
        if valid_df != None:
            valid_df.to_csv(fp_dai[2])
            valid = h2o_ai.create_dataset_sync(fp_dai[2])
            valid_key = valid.key
        else:
            valid_key = ''

        params = ModelParameters(dataset_key=train.key, testset_key=test.key,
                                 validset_key=valid_key, **model_params)
        exp = h2o_ai.start_experiment_sync(params)
        h2o_ai.download(src_path=exp.test_predictions_path, dest_dir='.')
        test_preds = pd.read_csv('./test_preds.csv')
        return exp, test_preds


    def _multiline(self):
        print ("Enter/Paste your content. 'end_query' to save it.")
        contents = []
        continue_query = True
        while continue_query:
            line = input("")
            if line == 'end_query':
                continue_query = False
                continue
            contents.append(line)
        return " ".join(contents)

    def _set_schema(self, col_name, col_type, mode):
        for i in range(len(col_name)):
            one_col = bigquery.SchemaField(col_name[i],
                                           col_type[i],
                                           mode=mode[i])
            self.SCHEMA.append(one_col)


if __name__ == "__main__":
    dataset = 'iris_dataset'
    pred_table = 'pred_table'
    bq_auth = '/Users/npng/Downloads/h2o-project-090347f40536.json'
    g_h2o = GoogleH2OIntegration(dataset, pred_table, bq_auth=bq_auth)
    df = g_h2o.bigquery_query()

    # NOTE: Do some feature engineering
    #       No feature engineering has been done for this toy example

    y = np.zeros((df.shape[0], 1))

    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2)
    df = df.drop(['Id','Species'], axis=1)

    # Initialize H2O and ingest data returned by bigquery query
    h2o.init(nthreads=-1, max_mem_size='2g', ip="127.0.0.1", port=54321)
    train_col = list(X_train.columns)
    test_col = list(X_test.columns)
    train = h2o.H2OFrame.from_python(X_train, column_names=train_col)
    test = h2o.H2OFrame.from_python(X_test, column_names=test_col)

    # Create lists of column names to be passed to H2O AutoML
    x = train.columns
    y = 'Species'
    ids = 'Id'
    x.remove(y)
    x.remove(ids)

    # Create instance of H2O AutoML and search for best model.
    aml = H2OAutoML(max_runtime_secs=30)
    aml.train(x=x, y=y, training_frame=train, leaderboard_frame=test)
    lb = aml.leaderboard
    print (lb)

    # Ingest data for prediction. Note that in this case I just used the same
    # data, In reality this would be new data.
    all_data = h2o.H2OFrame.from_python(df, column_names=x)
    predictions = aml.leader.predict(all_data).as_data_frame()

    # Zip together a test_id and predictions as list of tuples
    # The test_id's here as just toy ids generated by xrange()
    # [(test_id, prediction)] * n_rows
    # Call bigquery client to add contents of add_to_table to previously
    # created predictions_table
    test_ids = range(predictions.shape[0])
    g_h2o.write_to_table(test_ids, predictions['predict'])


    # TOY PARAMETERS
    # dataset = 'titanic_dataset'
    # client_args = {'address':'http://127.0.0.1:12345', 'username':'npng', 'password': 'npng'}
    # model_params = {'target_col':"Survived", 'cols_to_drop': ['PassengerId'],
    #                 'weight_col': None, "fold_col": None, 'time_col':'[AUTO]',
    #                 'is_classification':True, "enable_gpus":True, "seed": True,
    #                 'scorer':'auc', 'accuracy': 5, 'time':5, "interpretability":5}
    # bq_auth = '/Users/npng/Downloads/h2o-project-090347f40536.json'
    # pred_table = 'pred_table'
    # fp_local = ['~/data/train.csv', '~/data/test.csv']
    # fp_dai = ['/data/train.csv', '/data/test.csv']

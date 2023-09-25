from unittest import TestCase

import os

from rank_funcs import *
from pipelines import *


class Test(TestCase):
    def test_save_rankings(self):
        rankings = {"outlier": {"iqr": 100}, "missing": {"mean": 200}}
        directory = os.getcwd()
        filename = "//rankings//test_ranks.json"
        path = directory + filename
        error_code = save_rankings(rankings, path)
        file = open(path)
        data = json.load(file)
        file.close()
        assert (rankings == data)
        assert (error_code == 0)

    def test_save_rankings_fail(self):
        rankings = {"outlier": {"iqr": 100}, "missing": {"mean": 200}}
        path = "C://Users//chico//Desktop//MEI//Estágio//Simulator//thesis-project//rankings10//test_ranks.json"
        error_code = save_rankings(rankings, path)
        assert (error_code == 410)

    def test_get_rankings(self):
        rankings = {"outlier": {"iqr": 100}, "missing": {"mean": 200}}
        directory = os.getcwd()
        filename = "//rankings//test_ranks.json"
        path = directory + filename
        file = open(path, "w")
        file_json = json.dumps(rankings)
        file.write(file_json)
        file.close()
        rankings2, error_code = get_rankings(path)
        assert (rankings == rankings2)
        assert (error_code == 0)

    def test_get_rankings_fail(self):
        rankings = {"outlier": {"iqr": 100}, "missing": {"mean": 200}}
        directory = os.getcwd()
        filename = "//rankings10//test_ranks.json"
        path = directory + filename
        rankings2, error_code = get_rankings(path)
        assert (rankings2 is None)
        assert (error_code == 410)

    def test_pipeline(self):
        directory = os.getcwd()
        filename = "//rankings//"
        path = directory + filename
        id = 20001390
        df = pd.read_pickle(".//datasets//20001390.pkl")
        res, error_code = pre_processing_pipeline(df.values.tolist(), ["date", "BP", "HR"], id, path)
        nan = pd.isnull(res).any()
        assert (nan is not True)
        assert (error_code == 0)

    def test_pipeline_default_parameters(self):
        df = pd.read_pickle(".//datasets//20001390.pkl")
        res, error_code = pre_processing_pipeline(df.values.tolist(), "")
        nan = pd.isnull(res).any()
        assert (nan is not True)
        assert (error_code == 410)

    def test_create_miss(self):
        patterns = [

            {
                "incomplete_vars": [0],
                "mechanism": "MAR",
            },
            {
                "incomplete_vars": [1],
                "mechanism": "MAR",
            },
            {
                "incomplete_vars": [2],
                "mechanism": "MAR",
            },
            {
                "incomplete_vars": [3],
                "mechanism": "MAR",
            },

        ]
        df = pd.read_pickle(".//datasets//20001491.pkl")
        df.set_index('date', inplace=True)
        df_nan = create_missing_values_multivariate(df, 0.3, patterns)
        df_nan = pd.DataFrame(df_nan, columns=df.columns, index=df.index)
        df_nan[df_nan < 0] = np.nan

        nan = df_nan.isna().sum().sum()

        assert (nan <= df.shape[0])

    def test_pipeline_None(self):
        df = None
        res, error_code = pre_processing_pipeline(df, [])
        assert (error_code == 400)

    def test_pipeline_incorrect_type(self):
        df = "[1, 2, 3]"
        res, error_code = pre_processing_pipeline(df, [])
        assert (error_code == 401)

    def test_pipeline_training_None(self):
        df = None
        res, error_code = pre_processing_pipeline_training(df, [])
        assert (error_code == 400)

    def test_pipeline_training_incorrect_type(self):
        df = ""
        res, error_code = pre_processing_pipeline_training(df, [])
        assert (error_code == 401)

    def test_pipeline_training(self):
        directory = os.getcwd()
        filename = "//rankings//"
        path = directory + filename
        id = 20001491
        df = pd.read_pickle(".//datasets//20001491.pkl")
        cols = ["date", "BW", "BP", "HR", "BR"]
        df = df[cols]
        res, error_code = pre_processing_pipeline_training(df.values.tolist(), cols, id, path)
        assert (error_code == 0)

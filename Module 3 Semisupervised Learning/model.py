import ast
import re
from sklearn.preprocessing import RobustScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd


class KickstarterModel:

    def __init__(self):
        """
        Already implemented
        """

        self.model = None

    def preprocess_training_data(self, path_to_csv):
        """
        <Add your description here>
        """
        kick_train_data = pd.read_csv(path_to_csv)

        kick_train_data.drop(
            ['staff_pick', 'backers_count'],
            inplace=True,
            axis=1)
        y = (kick_train_data['state'] == "successful").astype(int)
        kick_train_data.drop(['state'], inplace=True, axis=1)

        # replace blurb by length of blurb
        lengthofblurb = kick_train_data['blurb'].apply(lambda x: len(str(x)))
        kick_train_data['lengthofblurb'] = lengthofblurb
        kick_train_data.drop(['blurb'], inplace=True, axis=1)

        subcategory = []
        for entry in kick_train_data['category']:
            my_dict = ast.literal_eval(entry)
            slugvalue = my_dict['slug']
            subcategory.append(slugvalue.split('/')[1])

        kick_train_data['subcategory'] = subcategory
        kick_train_data.drop(['category'], inplace=True, axis=1)

        blurbinprofile = []
        for entry in kick_train_data['profile']:
            entry = entry.replace('"', '')
            foundblurb = re.search("blurb:(.*?),", entry)
            blurbinprofile.append(foundblurb.group(1))
        blurbinprofile_one_zero = [0 if i == 'null' else 1 for i in blurbinprofile]
        kick_train_data['blurbinprofile'] = blurbinprofile_one_zero

        goalinusdollars = (kick_train_data['goal'] *
                           kick_train_data['static_usd_rate'])
        kick_train_data['goalinusdollars'] = goalinusdollars
        kick_train_data.drop(['goal'], inplace=True, axis=1)

        kick_train_data = pd.get_dummies(
            data=kick_train_data,
            columns=['country'],
            prefix=['country'])

        kick_train_data = pd.get_dummies(
            data=kick_train_data,
            columns=['subcategory'],
            prefix=['subcategory'])

        kick_train_data['deadline'] = pd.to_datetime(
            kick_train_data['deadline'],
            origin='unix',
            unit='s')
        kick_train_data['created_at'] = pd.to_datetime(
            kick_train_data['created_at'],
            origin='unix',
            unit='s')
        kick_train_data['launched_at'] = pd.to_datetime(
            kick_train_data['launched_at'],
            origin='unix',
            unit='s')

        # deal with dates part 2
        # diff between deadline and launch
        openforfundingdiff = (kick_train_data['deadline'] -
                              kick_train_data['launched_at'])
        # give me my diffindayscolumn to add to dataframe
        openforfundingindays = openforfundingdiff.apply(
            lambda x: x.days)

        # diff between deadline and launch
        createdtolauncheddiff = (kick_train_data['launched_at'] -
                                 kick_train_data['created_at'])
        # give me my diffindayscolumn to add to dataframe
        ctoldiffindays = createdtolauncheddiff.apply(
            lambda x: x.days)

        kick_train_data['openforfundingindays'] = openforfundingindays
        kick_train_data['createdtolauncheddiffindays'] = ctoldiffindays

        kick_train_data.drop(['deadline'], inplace=True, axis=1)
        kick_train_data.drop(['created_at'], inplace=True, axis=1)
        kick_train_data.drop(['launched_at'], inplace=True, axis=1)
        kick_train_data.drop(['state_changed_at'], inplace=True, axis=1)

        # drop columns I said I should because they are no good
        kick_train_data.drop(['Unnamed: 0'], inplace=True, axis=1)
        kick_train_data.drop(['id'], inplace=True, axis=1)
        kick_train_data.drop(['photo'], inplace=True, axis=1)
        kick_train_data.drop(['name'], inplace=True, axis=1)
        kick_train_data.drop(['slug'], inplace=True, axis=1)
        kick_train_data.drop(['currency'], inplace=True, axis=1)
        kick_train_data.drop(['currency_symbol'], inplace=True, axis=1)
        kick_train_data.drop(['currency_trailing_code'], inplace=True, axis=1)
        kick_train_data.drop(['creator'], inplace=True, axis=1)
        kick_train_data.drop(['location'], inplace=True, axis=1)
        kick_train_data.drop(['profile'], inplace=True, axis=1)
        kick_train_data.drop(['urls'], inplace=True, axis=1)
        kick_train_data.drop(['source_url'], inplace=True, axis=1)
        kick_train_data.drop(['friends'], inplace=True, axis=1)
        kick_train_data.drop(['is_starred'], inplace=True, axis=1)
        kick_train_data.drop(['is_backing'], inplace=True, axis=1)
        kick_train_data.drop(['permissions'], inplace=True, axis=1)
        kick_train_data.drop(['static_usd_rate'], inplace=True, axis=1)
        kick_train_data.drop(['disable_communication'], inplace=True, axis=1)

        rb_lengthofblurb = (RobustScaler(quantile_range=(25, 75))
                            .fit_transform(
                                kick_train_data['lengthofblurb']
                                           .reshape(-1, 1)))
        rb_goalinusdollars = (RobustScaler(quantile_range=(25, 75))
                              .fit_transform(
                                  kick_train_data['goalinusdollars']
                                             .reshape(-1, 1)))
        rb_openforfundingindays = (RobustScaler(quantile_range=(25, 75))
                                   .fit_transform(
                                       kick_train_data['openforfundingindays']
                                                  .reshape(-1, 1)))
        rb_createdtolauncheddiffindays = (RobustScaler(quantile_range=(25, 75))
                                          .fit_transform(
                                              kick_train_data['createdtolauncheddiffindays']
                                              .reshape(-1, 1)))

        kick_train_data['rb_lengthofblurb'] = rb_lengthofblurb
        kick_train_data['rb_goalinusdollars'] = rb_goalinusdollars
        kick_train_data['rb_openforfundingindays'] = rb_openforfundingindays
        kick_train_data['rb_createdtolauncheddiffindays'] = (
            rb_createdtolauncheddiffindays)

        kick_train_data.drop(['lengthofblurb'], inplace=True, axis=1)
        kick_train_data.drop(['goalinusdollars'], inplace=True, axis=1)
        kick_train_data.drop(['openforfundingindays'],
                             inplace=True,
                             axis=1)
        kick_train_data.drop(['createdtolauncheddiffindays'],
                             inplace=True,
                             axis=1)

        X = kick_train_data.copy(deep=True)

        return X, y

    def fit(self, X, y):
        """
        <Add your description here>
        """
        lda = LinearDiscriminantAnalysis(solver='lsqr')
        lda.fit(X, y)

        self.model = lda

    def preprocess_unseen_data(self, path_to_csv):
        """
        <Add your description here>
        """
        kick_train_data = pd.read_csv(path_to_csv)

        kick_train_data.drop(['staff_pick', 'backers_count'],
                             inplace=True,
                             axis=1)

        # replace blurb by length of blurb
        lengthofblurb = kick_train_data['blurb'].apply(
            lambda x: len(str(x)))
        kick_train_data['lengthofblurb'] = lengthofblurb
        kick_train_data.drop(['blurb'], inplace=True, axis=1)

        subcategory = []
        for entry in kick_train_data['category']:
            my_dict = ast.literal_eval(entry)
            slugvalue = my_dict['slug']
            subcategory.append(slugvalue.split('/')[1])

        kick_train_data['subcategory'] = subcategory
        kick_train_data.drop(['category'], inplace=True, axis=1)

        blurbinprofile = []
        for entry in kick_train_data['profile']:
            entry = entry.replace('"', '')
            foundblurb = re.search("blurb:(.*?),", entry)
            blurbinprofile.append(foundblurb.group(1))
        blurbinprofile_one_zero = (
            [0 if i == 'null' else 1 for i in blurbinprofile])
        kick_train_data['blurbinprofile'] = blurbinprofile_one_zero

        goalinusdollars = (kick_train_data['goal'] *
                           kick_train_data['static_usd_rate'])
        kick_train_data['goalinusdollars'] = goalinusdollars
        kick_train_data.drop(['goal'], inplace=True, axis=1)

        kick_train_data = pd.get_dummies(data=kick_train_data,
                                         columns=['country'],
                                         prefix=['country'])

        kick_train_data = pd.get_dummies(data=kick_train_data,
                                         columns=['subcategory'],
                                         prefix=['subcategory'])

        kick_train_data['deadline'] = pd.to_datetime(
            kick_train_data['deadline'],
            origin='unix',
            unit='s')
        kick_train_data['created_at'] = pd.to_datetime(
            kick_train_data['created_at'],
            origin='unix',
            unit='s')
        kick_train_data['launched_at'] = pd.to_datetime(
            kick_train_data['launched_at'],
            origin='unix',
            unit='s')

        # deal with dates part 2
        # diff between deadline and launch
        openforfundingdiff = (kick_train_data['deadline'] -
                              kick_train_data['launched_at'])
        # give me my diffindayscolumn to add to dataframe
        openforfundingindays = openforfundingdiff.apply(
            lambda x: x.days)

        # diff between deadline and launch
        createdtolauncheddiff = (kick_train_data['launched_at'] -
                                 kick_train_data['created_at'])
        # give me my diffindayscolumn to add to dataframe
        ctoldiffindays = createdtolauncheddiff.apply(
            lambda x: x.days)

        kick_train_data['openforfundingindays'] = openforfundingindays
        kick_train_data['createdtolauncheddiffindays'] = ctoldiffindays

        kick_train_data.drop(['deadline'], inplace=True, axis=1)
        kick_train_data.drop(['created_at'], inplace=True, axis=1)
        kick_train_data.drop(['launched_at'], inplace=True, axis=1)
        kick_train_data.drop(['state_changed_at'],
                             inplace=True,
                             axis=1)

        # drop columns I said I should because they are no good
        kick_train_data.drop(['Unnamed: 0'],
                             inplace=True,
                             axis=1)
        kick_train_data.drop(['id'], inplace=True, axis=1)
        kick_train_data.drop(['photo'], inplace=True, axis=1)
        kick_train_data.drop(['name'], inplace=True, axis=1)
        kick_train_data.drop(['slug'], inplace=True, axis=1)
        kick_train_data.drop(['currency'], inplace=True, axis=1)
        kick_train_data.drop(['currency_symbol'],
                             inplace=True,
                             axis=1)
        kick_train_data.drop(['currency_trailing_code'],
                             inplace=True,
                             axis=1)
        kick_train_data.drop(['creator'], inplace=True, axis=1)
        kick_train_data.drop(['location'], inplace=True, axis=1)
        kick_train_data.drop(['profile'], inplace=True, axis=1)
        kick_train_data.drop(['urls'], inplace=True, axis=1)
        kick_train_data.drop(['source_url'], inplace=True, axis=1)
        kick_train_data.drop(['friends'], inplace=True, axis=1)
        kick_train_data.drop(['is_starred'], inplace=True, axis=1)
        kick_train_data.drop(['is_backing'], inplace=True, axis=1)
        kick_train_data.drop(['permissions'], inplace=True, axis=1)
        kick_train_data.drop(['static_usd_rate'], inplace=True, axis=1)
        kick_train_data.drop(['disable_communication'], inplace=True, axis=1)

        rb_lengthofblurb = RobustScaler(
            quantile_range=(25, 75)).fit_transform(
            kick_train_data['lengthofblurb']
            .values.reshape(-1, 1))
        rb_goalinusdollars = RobustScaler(
            quantile_range=(25, 75)).fit_transform(
            kick_train_data['goalinusdollars']
            .values.reshape(-1, 1))
        rb_openforfundingindays = RobustScaler(
            quantile_range=(25, 75)).fit_transform(
            kick_train_data['openforfundingindays']
            .values.reshape(-1, 1))
        rb_ctldiffindays = RobustScaler(
            quantile_range=(25, 75)).fit_transform(
            kick_train_data['createdtolauncheddiffindays']
            .values.reshape(-1, 1))

        kick_train_data['rb_lengthofblurb'] = rb_lengthofblurb
        kick_train_data['rb_goalinusdollars'] = rb_goalinusdollars
        kick_train_data['rb_openforfundingindays'] = rb_openforfundingindays
        kick_train_data['rb_createdtolauncheddiffindays'] = rb_ctldiffindays

        kick_train_data.drop(['lengthofblurb'], inplace=True, axis=1)
        kick_train_data.drop(['goalinusdollars'], inplace=True, axis=1)
        kick_train_data.drop(['openforfundingindays'], inplace=True, axis=1)
        kick_train_data.drop(['createdtolauncheddiffindays']
                            , inplace=True
                            , axis=1)

        X = kick_train_data.copy(deep=True)

        return X

    def predict(self, X):
        """
        predict the lda
        """

        return self.model.predict(X)

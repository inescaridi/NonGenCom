from typing import List, Optional

from pandas import DataFrame

from nonGenCom.Utils import merge_dbs
from nonGenCom.Variables.Variable import Variable


class ScoreCalculator:
    def __init__(self, fc_db: DataFrame, mp_db: DataFrame, fc_index_colname: str, mp_index_colname: str,
                 fc_elements_id: Optional[List] = None):
        """
        :param fc_db: Forensic Case Database
        :param mp_db: Missing Person Database
        :param fc_index_colname: colname of Fosensic Case ID
        :param mp_index_colname: colname of Missing Person ID

        :param fc_elements_id: Optional[List]: if None is passed then uses full fc_db for FC-selection comparison,
        otherwise filter by IDs included in fc_elements_id
        """

        self.merged_dbs = merge_dbs(fc_db, mp_db, fc_index_colname, '_FC', '_MP', fc_elements_id)

        if fc_index_colname == mp_index_colname:  # there can be a collision on merge
            fc_index_colname += '_FC'
            mp_index_colname += '_MP'

        self.fc_index_colname = fc_index_colname
        self.mp_index_colname = mp_index_colname

    def add_score(self, variable: Variable, context_name: str, scenery_name: str,
                  fc_value_colname: str, mp_value_colname: str) -> DataFrame:
        """
        # TODO complete docstring

        :param variable: Variable to calculate
        :param context_name:
        :param scenery_name:

        :param fc_value_colname: colname of value for variable in Fosensic Case Database
        :param mp_value_colname: colname of value for variable in Missing Person Database

        :return:
        """
        # TODO move all database "config" (column names mostly) to a class
        posterior = variable.get_posterior(context_name, scenery_name)
        print(f"Context: {context_name}")
        print(f"Scenery: {scenery_name}")
        print("Posterior\n", posterior, "\n")

        if fc_value_colname == mp_value_colname:  # there can be a collision on merge
            fc_value_colname += '_FC'
            mp_value_colname += '_MP'

        # create new FC and MP columns according to the renaming dict
        merged_dbs = self.merged_dbs
        merged_dbs['fc_index_aux'] = merged_dbs[fc_value_colname].map(variable.renames)
        merged_dbs['mp_index_aux'] = merged_dbs[mp_value_colname].map(variable.renames)

        # set the same index as posterior
        merged_dbs = merged_dbs.set_index(['fc_index_aux', 'mp_index_aux'])
        merged_dbs.index = merged_dbs.index.rename(['FC', 'MP'])

        # merge with posterior
        merged_dbs = merged_dbs.join(posterior.rename(variable.score_column_name))\
            .reset_index(drop=True)\
            .sort_values(variable.score_column_name, ascending=False)

        self.merged_dbs = merged_dbs
        return merged_dbs

    def get_results(self):
        return self.merged_dbs

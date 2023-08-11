from abc import ABC, abstractmethod

from pandas import Series

from nonGenCom.Variable import Variable


class ContinuousVariable(Variable, ABC):
    def __init__(self, contexts_path: str | None, fc_sceneries_path: str | None, mp_sceneries_path: str | None,
                 context_name: str | None, fc_scenery_name: str | None, mp_scenery_name: str | None,
                 min_value, max_value, step):
        self.min_value = min_value
        self.max_value = max_value
        self.step = step
        super().__init__(contexts_path, fc_sceneries_path, mp_sceneries_path, context_name, fc_scenery_name,
                         mp_scenery_name)

    def get_fc_likelihood(self, scenery_name: str) -> Series:
        pass

    def get_fc_score(self) -> Series:
        pass

    def get_mp_likelihood(self, scenery_name: str) -> Series:
        pass

    def get_mp_score(self, context_name: str, scenery_name: str) -> Series:
        pass

    @abstractmethod
    def _get_fc_likelihood_for_combination(self, mp_category: tuple, fc_category: tuple):
        raise NotImplementedError()

    @abstractmethod
    def _get_mp_likelihood_for_combination(self, mp_category: tuple, fc_category: tuple):
        raise NotImplementedError()

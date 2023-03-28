import pathlib
import unittest

from classes.commonUtils import CommonUtils


class TestUtils(unittest.TestCase):
    @staticmethod
    def getTestFolder():
        solutionFolder = CommonUtils.getSolutionFolder()
        return pathlib.Path(solutionFolder).joinpath("Framework").joinpath("test").resolve()

    @staticmethod
    def getResources():
        solutionFolder = CommonUtils.getSolutionFolder()
        return pathlib.Path(solutionFolder).joinpath("Framework").joinpath("test").joinpath("results").resolve()

    @staticmethod
    def getMainResourcesFolder():
        return CommonUtils.getMainResourcesFolder()

    @staticmethod
    def getMainResourcesIMDBFolder():
        return CommonUtils.getMainResourcesIMDBFolder()

    @staticmethod
    def getIMDBWithVectorsFolder():
        return "D:\PYTHON\Programms\GrantOfPresident2022\HumanLanguageProject\IMDB"

    @staticmethod
    def getIMDBWithVectorsFolderFull():
        return "D:\Vadim\PYTHON\Programms\PresidentGrant2022_2023\IMDB"

    @staticmethod
    def getSaveWeightsFolder():
        solutionFolder = CommonUtils.getSolutionFolder()
        return pathlib.Path(solutionFolder).joinpath("Framework").joinpath("test").joinpath(
            "results").joinpath(
            "weights").resolve()
if __name__ == "__main__":
    unittest.main()

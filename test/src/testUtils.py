import pathlib
import unittest

from classes.commonUtils import CommonUtils


class TestUtils(unittest.TestCase):
    @staticmethod
    def getTestFolder():
        solutionFolder = CommonUtils.getSolutionFolder()
        return pathlib.Path(solutionFolder).joinpath("ImageProcessingHandmade").joinpath("test").resolve()

    @staticmethod
    def getResources():
        solutionFolder = CommonUtils.getSolutionFolder()
        return pathlib.Path(solutionFolder).joinpath("ImageProcessingHandmade").joinpath("test").joinpath("results").resolve()

    @staticmethod
    def getMainResourcesFolder():
        return CommonUtils.getMainResourcesFolder()

    @staticmethod
    def getMainResourcesIMDBFolder():
        return CommonUtils.getMainResourcesIMDBFolder()

if __name__ == "__main__":
    unittest.main()

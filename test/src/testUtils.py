import pathlib
import unittest

from common.commonUtils import CommonUtils


class TestUtils(unittest.TestCase):
    @staticmethod
    def getTestFolder():
        solutionFolder = CommonUtils.getSolutionFolder()
        return pathlib.Path(solutionFolder).joinpath("ImageProcessingHandmade").joinpath("test").resolve()

    @staticmethod
    def getResources():
        solutionFolder = CommonUtils.getSolutionFolder()
        return pathlib.Path(solutionFolder).joinpath("ImageProcessingHandmade").joinpath("test").joinpath("resultsData").resolve()

    @staticmethod
    def getMainResourcesFolder():
        return CommonUtils.getMainResourcesFolder()

    @staticmethod
    def getMainResourcesXPMFolder():
        return CommonUtils.getMainResourcesXPMFolder()
    @staticmethod
    def getMultiprocessing():
        solutionFolder = CommonUtils.getSolutionFolder()
        return pathlib.Path(solutionFolder).joinpath("ImageProcessingHandmade").joinpath("main").joinpath("src").joinpath("imageScaleParallel").resolve()

    @staticmethod
    def getMultiprocessingSimple():
        solutionFolder = CommonUtils.getSolutionFolder()
        return pathlib.Path(solutionFolder).joinpath("ImageProcessingHandmade").joinpath("main").joinpath(
            "src").joinpath("simpleExample").resolve()

    @staticmethod
    def getMultiprocessingConvolutionNN():
        solutionFolder = CommonUtils.getSolutionFolder()
        return pathlib.Path(solutionFolder).joinpath("ImageProcessingHandmade").joinpath("main").joinpath(
            "src").joinpath("convolutionNN").resolve()

if __name__ == "__main__":
    unittest.main()

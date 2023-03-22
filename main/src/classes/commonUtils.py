import pathlib


class CommonUtils(object):
    @staticmethod
    def getSolutionFolder():
        return pathlib.Path(__file__).parent.parent.parent.parent.parent.resolve()

    @staticmethod
    def getProjectFolder():
        solutionFolder = CommonUtils.getSolutionFolder()
        return pathlib.Path(solutionFolder).joinpath("ImageProcessingHandmade").joinpath("main").resolve()

    @staticmethod
    def getMainResourcesFolder():
        solutionFolder = CommonUtils.getSolutionFolder()
        return pathlib.Path(solutionFolder).joinpath("ImageProcessingHandmade").joinpath("main").joinpath("resources").resolve()

    @staticmethod
    def getMainResourcesXPMFolder():
        solutionFolder = CommonUtils.getSolutionFolder()
        return pathlib.Path(solutionFolder).joinpath("ImageProcessingHandmade").joinpath("main").joinpath("resources").joinpath("xpm").resolve()
import pathlib


class CommonUtils(object):
    @staticmethod
    def getSolutionFolder():
        return pathlib.Path(__file__).parent.parent.parent.parent.parent.resolve()

    @staticmethod
    def getProjectFolder():
        solutionFolder = CommonUtils.getSolutionFolder()
        return pathlib.Path(solutionFolder).joinpath("Framework").joinpath("main").resolve()

    @staticmethod
    def getMainResourcesFolder():
        solutionFolder = CommonUtils.getSolutionFolder()
        return pathlib.Path(solutionFolder).joinpath("Framework").joinpath("main").joinpath("resources").resolve()
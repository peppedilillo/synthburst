from pathlib import Path

prjpath = Path(__file__).parent
assets = prjpath.joinpath("assets")
database = assets.joinpath("gbmburstcatalog.csv.zip")


def create_if_not_exists(func):
    def wrapper(*args):
        p = func(*args)
        p.mkdir(exist_ok=True)
        return p

    return wrapper


@create_if_not_exists
def ttes() -> Path:
    return prjpath.joinpath("ttedata")

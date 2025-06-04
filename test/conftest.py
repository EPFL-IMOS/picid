# Important to set the environment variable PROJECT_ROOT to the root of the project, hence always import lmetk first!
import lmetk # noqa: F401
import sys
import os

print(os.environ['PROJECT_ROOT']) 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def pytest_addoption(parser):
    parser.addoption("--paths", action="store", default="default", help="A hydra paths config name to localize testing.")
from Executor import Executor
from SecondLevelLanguageParser import Parser

path = f'test.txt'
with open(path, 'r', encoding='utf-8') as f:
    create_test = f.readlines()
testPar = Parser(create_test)
result = testPar()
executor = Executor(result)
executor.run()

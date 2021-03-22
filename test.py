import re
reg = 'x ([a-z])|y ([a-z])'
ma = re.match(reg, 'y s')
if ma:
    print(ma.group(2))
else:
    print(None)
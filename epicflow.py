import sys
sys.path.append( '/home/erhsin/lib' )

import matlab.engine as mateng

matlab_eng = mateng.start_matlab()
matlab_eng.addpath('./matlab')
out = matlab_eng.epicflow('../1.jpg', '../2.jpg')
print out.split('$')

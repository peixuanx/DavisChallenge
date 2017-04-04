import sys
sys.path.append( '/home/erhsin/lib' )
import matlab.engine

matlab_eng = matlab.engine.start_matlab()
out = matlab_eng.epicflow('../1.jpg', '../2.jpg')
print out.split('$')

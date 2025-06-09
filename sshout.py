#Returns file instead of print for testing results to be available if ssh is terminated 
import os 

def fileout(filename): #filename must be str 
    with o as os.file('r', filename):
        

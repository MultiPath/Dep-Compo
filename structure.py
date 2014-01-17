#/usr/bin/python2.7
#*-coding:utf-8-*-
from RAE import RAE
import numpy as np
import re

#Describe the structure of Dependency Tree.
class DependencyTree(object):
	def __init__(self):
		self.root  = SubTree('ROOT', 0)
		self.group = []
		self.sent  = ''
		pass

	def read_sent(self, lineb):
		t1 = 0; t2 = 0
		words = [('ROOT','ROOT')];
		for i in xrange(len(lineb)):
			c = lineb[i]; 
			if c == '(':
				t1 = i
			elif c == ')':
				t2 = i
				if t1 != -1:    
					z =  lineb[t1+1:t2]
					zz = z.split(' ')
					words.append((zz[1],zz[0]))
					t1 = -1
		return words
			
	def read_dependency(self, lineb):
		pat    = re.compile('\S+\(\S+\,\s*\S+\)')
		group  = pat.findall(lineb[1:-1])
		#print lineb
		#print group
		result = []
		
		for g in group:
			g = g.split('(')
			relation = g[0]
			#print g
			pat2 = re.compile('[^\s\,]+\-\d+')
			gro2 = pat2.findall(g[1])
			#print gro2	
			parent  = gro2[0]
			child   = gro2[1]
			
			#print parent, child, relation
			ppos   = pos2float(parent[parent.rfind('-')+1:])
			parent = parent[0: parent.rfind('-')].strip()
			
			cpos   = pos2float(child[child.rfind('-')+1:])
			child  = child[0: child.rfind('-')].strip()
			
			result.append((parent, ppos, child, cpos, relation))
		return result
	
	def build(self, subtree):
		pass

#Function: help change pos-info into float.
def pos2float(pos):
	tmp = 0
	while pos[-1] == '\'':
		tmp += 0.1
		pos = pos[0:-1]
	return float(pos)+tmp

#Describe the basic structure of a sub-tree.
class SubTree(object):
	def __init__(self, name, pos):
		self.name   = name
		self.pos    = pos

		self.vector = None

		self.left   = []
		self.right  = []
		pass


		



		







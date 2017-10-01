class AttribFeature(object):

	""" Name is the name of the feature, attrib is the column of the df that has it
		Features straight from the dataframe
	"""

	def __init__(self, name, attrib):	
		self.name = name
		self.attrib = attrib



class CalcFeature(object):


	""" Engineered Features. func_one is how to change one cell into a feature
		func_many is for a whole column
	"""

	def __init__(self, name, func_one, func_many):
		self.name = name
		self.func_one = func_one
		self.func_many = func_many

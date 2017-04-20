

def outputDim(indim, depth):
	temp = indim
	# down
	for i in range(depth):
		temp = temp/2 -2
	# up
	temp = temp - 4
	for i in range(depth):
		temp = temp*2 -4
	return temp


def main():
	indim = 768
	depth = 3
	outdim = outputDim(indim,depth) 
	print(outdim)
	print("NUM of pixels= " + str(float((indim-outdim)/2)/indim))


if __name__ == '__main__':
	main()
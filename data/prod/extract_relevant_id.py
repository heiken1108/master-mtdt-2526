import pandas as pd
from utils import file

def get_collection_ids():
	df = file.load_collection_data("Collection_data.csv")
	ids = df["PersonId"].unique()
	return ids

def assess_line(line, ids):
	pId = int(line.split(";")[2])
	if pId in ids:
		#print(pId)
		return True
	else:
		return False
	

def extract_lines(input_file):
	ids = get_collection_ids()
	lines = []
	no = 0
	with open(input_file, 'r', encoding='utf-8') as f:
		for line in f:
			if no % 100000 == 0:
				print(no)
			line = line.strip()
			if assess_line(line, ids):
				lines.append(line)
			no = no+1
	return lines

def write_lines_to_file(lines, output_file):
	with open(output_file, "w", encoding="utf-8") as f:
		f.write("\n".join(lines) + "\n")

if __name__ == "__main__":
	input_file = "konto_data.csv"
	output_file = "konto_data_trimmed.csv"
	lines = extract_lines(input_file)
	write_lines_to_file(lines, output_file)
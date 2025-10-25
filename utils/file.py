import pandas as pd

def load_collection_data(file_path):
	collection_frame = pd.read_csv(file_path, sep=';', decimal=',')
	d_datecols = ['CollectionClosedDateId', 'CollectionOpenedDateId']
	m_datecols = ['registreringsmaaned']
	for col in d_datecols:
		collection_frame[col] = pd.to_datetime(collection_frame[col].astype('Int64'), format='%Y%m%d', errors='coerce').dt.strftime('%d-%m-%Y')
	for col in m_datecols:
		collection_frame[col] = pd.to_datetime(collection_frame[col].astype('Int64'), format='%Y%m', errors='coerce').dt.strftime('%m-%Y')
	collection_frame = collection_frame.sort_values(by=["PersonId", "yearmonth"], ascending=True)
	return collection_frame

def load_konto_data(file_path):
	duplicate_cols = [43, 44]
	konto_frame = pd.read_csv(file_path, usecols=[i for i in range(102) if i not in duplicate_cols], decimal=',')
	konto_frame = konto_frame.sort_values(by=['PersonId', 'YearMonth'], ascending=True)
	return konto_frame

def print_n_raw_lines(file_path, n_lines=1):
	if n_lines < 1:
		return
	with open(file_path, 'r', encoding='utf-8') as f:
		for i in range(n_lines):
			line = f.readline()
			print(line)
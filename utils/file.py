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
	konto_frame = pd.read_csv(file_path, sep=';', usecols=[i for i in range(102) if i not in duplicate_cols], decimal=',')

	#Add common value if any NaNs
	for col in ["Gender"]:
		konto_frame[col] = konto_frame.groupby("PersonId")[col].transform(lambda x: x.ffill().bfill())

	#Strip columns
	for col in ["AgeGroup2", "Gender"]:
		konto_frame[col] = konto_frame[col].astype(str).str.strip()

	#Rename columns
	konto_frame.rename(columns={"AgeGroup2": "AgeGroup"}, inplace=True)
	konto_frame.rename(columns={"kommunenavn": "Kommunenavn"}, inplace=True)

	#Drop columns
	cols_to_drop = ["kommunenr", "GeneralStatus"]
	konto_frame.drop(columns=cols_to_drop, inplace=True)

	#Fill nan Kommunenavn with Ukjent
	konto_frame["Kommunenavn"].fillna("UKJENT", inplace=True)

	#Convert from float to int
	konto_frame["CreditLimitAmt"] = konto_frame["CreditLimitAmt"].astype(int)

	#Convert NaN to 0
	nan_to_0_cols = ['SumL12_Airlines', 'SumL3_Airlines', 'Last_Airlines', 'SumL12_Amusement and Entertainment', 'SumL3_Amusement and Entertainment', 'Last_Amusement and Entertainment', 'SumL12_Automobile / Vehicle Rental', 'SumL3_Automobile / Vehicle Rental', 'Last_Automobile / Vehicle Rental', 'SumL12_Business Services', 'SumL3_Business Services', 'Last_Business Services', 'SumL12_Clothing Stores', 'SumL3_Clothing Stores', 'Last_Clothing Stores', 'SumL12_Contracted Services', 'SumL3_Contracted Services', 'Last_Contracted Services', 'SumL12_Government Services', 'SumL3_Government Services', 'Last_Government Services', 'SumL12_Hotels', 'SumL3_Hotels', 'Last_Hotels', 'SumL12_Includes all lodging merchants', 'SumL3_Includes all lodging merchants', 'Last_Includes all lodging merchants', 'SumL12_Mail Order / Telephone Order Providers', 'SumL3_Mail Order / Telephone Order Providers', 'Last_Mail Order / Telephone Order Providers', 'SumL12_Miscellaneous Stores', 'SumL3_Miscellaneous Stores', 'Last_Miscellaneous Stores', 'SumL12_Others', 'SumL3_Others', 'Last_Others', 'SumL12_Professional Services and Membership Organizations', 'SumL3_Professional Services and Membership Organizations', 'Last_Professional Services and Membership Organizations', 'SumL12_Repair Services', 'SumL3_Repair Services', 'Last_Repair Services', 'SumL12_Retail Stores', 'SumL3_Retail Stores', 'Last_Retail Stores', 'SumL12_Service Providers', 'SumL3_Service Providers', 'Last_Service Providers', 'SumL12_Transportation', 'SumL3_Transportation', 'Last_Transportation', 'SumL12_Utilities', 'SumL3_Utilities', 'Last_Utilities', 'SumL12_Wholesale Distributors and Manufacturers', 'SumL3_Wholesale Distributors and Manufacturers', 'Last_Wholesale Distributors and Manufacturers'] 
	for col in nan_to_0_cols:
		konto_frame[col].fillna(0, inplace=True)

	konto_frame = konto_frame.sort_values(by=['PersonId', 'YearMonth'], ascending=True)
	return konto_frame

def print_n_raw_lines(file_path, n_lines=1):
	if n_lines < 1:
		return
	with open(file_path, 'r', encoding='utf-8') as f:
		for i in range(n_lines):
			line = f.readline()
			print(line)
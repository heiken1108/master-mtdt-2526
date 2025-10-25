konto_behave_cols = ['BalanceAmt', 'InterestEarningLendingAmt', 'CreditLimitAmt', 'CreditLimitInceaseFlag', 'CollectionFlag', 'ClosedAccountFlag', 'TurnoverAmt', 'TurnoverNum', 'TurnoverDomAmt', 'TurnoverDomNum', 'TurnoverIntAmt', 'TurnoverIntNum', 'FundtransferAmt', 'FundtransferNum', 'CashAtmAmt', 'CashAtmNum', 'CashCounterAmt', 'CashCounterNum', 'OverdueAmt', 'StatementClosingBalanceAmt', 'StatementEffectivePaymentsAmt', 'StatementMinimumToPayAmt', 'PaymentOverDueFlag', 'FullpayerFlag', 'RevolvingFlag', 'ActiveFlag', 'CardUsedFlag', 'TransactionFlag', 'Last_Airlines', 'Last_Amusement and Entertainment', 'Last_Automobile / Vehicle Rental', 'Last_Business Services', 'Last_Clothing Stores', 'Last_Contracted Services', 'Last_Government Services', 'Last_Hotels', 'Last_Includes all lodging merchants', 'Last_Mail Order / Telephone Order Providers', 'Last_Miscellaneous Stores', 'Last_Others', 'Last_Professional Services and Membership Organizations', 'Last_Repair Services', 'Last_Retail Stores', 'Last_Service Providers', 'Last_Transportation', 'Last_Utilities', 'Last_Wholesale Distributors and Manufacturers']
konto_trans_cols = ['Last_Airlines', 'Last_Amusement and Entertainment', 'Last_Automobile / Vehicle Rental', 'Last_Business Services', 'Last_Clothing Stores', 'Last_Contracted Services', 'Last_Government Services', 'Last_Hotels', 'Last_Includes all lodging merchants', 'Last_Mail Order / Telephone Order Providers', 'Last_Miscellaneous Stores', 'Last_Others', 'Last_Professional Services and Membership Organizations', 'Last_Repair Services', 'Last_Retail Stores', 'Last_Service Providers', 'Last_Transportation', 'Last_Utilities', 'Last_Wholesale Distributors and Manufacturers']
konto_profile_cols = ['YearMonth', 'PersonId', 'AccountId', 'ProductId', 'DistributorId', 'AgeGroup2', 'Gender', 'kommunenavn', 'MonthsSinceAccountCreatedNum', 'CreditLimitAmt']

def konto_behavioural_data(konto_frame, dict=False):
	behave_df = konto_frame[konto_behave_cols]
	if not dict:
		return behave_df

def konto_transactional_data(konto_frame, dict=False):
	trans_df = konto_frame[konto_trans_cols]
	if not dict:
		return trans_df

def konto_profile_data(konto_frame):
	profile_df = konto_frame[konto_profile_cols]
	return profile_df
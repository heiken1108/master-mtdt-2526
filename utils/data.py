import pandas as pd

_konto_behave_cols = [
    "PersonId",
    "AccountId",
    "YearMonth",
    "BalanceAmt",
    "InterestEarningLendingAmt",
    "CreditLimitAmt",
    "CreditLimitIncreaseFlag",
    "CollectionFlag",
    "ClosedAccountFlag",
    "TurnoverAmt",
    "TurnoverNum",
    "TurnoverDomAmt",
    "TurnoverDomNum",
    "TurnoverIntAmt",
    "TurnoverIntNum",
    "FundtransferAmt",
    "FundtransferNum",
    "CashAtmAmt",
    "CashAtmNum",
    "CashCounterAmt",
    "CashCounterNum",
    "OverdueAmt",
    "StatementClosingBalanceAmt",
    "StatementEffectivePaymentsAmt",
    "StatementMinimumToPayAmt",
    "PaymentOverDueFlag",
    "FullpayerFlag",
    "RevolvingFlag",
    "ActiveFlag",
    "CardUsedFlag",
    "TransactionFlag",
    "Last_Airlines",
    "Last_Amusement and Entertainment",
    "Last_Automobile / Vehicle Rental",
    "Last_Business Services",
    "Last_Clothing Stores",
    "Last_Contracted Services",
    "Last_Government Services",
    "Last_Hotels",
    "Last_Includes all lodging merchants",
    "Last_Mail Order / Telephone Order Providers",
    "Last_Miscellaneous Stores",
    "Last_Others",
    "Last_Professional Services and Membership Organizations",
    "Last_Repair Services",
    "Last_Retail Stores",
    "Last_Service Providers",
    "Last_Transportation",
    "Last_Utilities",
    "Last_Wholesale Distributors and Manufacturers",
]
_konto_trans_cols = [
    "PersonId",
    "AccountId",
    "YearMonth",
    "Last_Airlines",
    "Last_Amusement and Entertainment",
    "Last_Automobile / Vehicle Rental",
    "Last_Business Services",
    "Last_Clothing Stores",
    "Last_Contracted Services",
    "Last_Government Services",
    "Last_Hotels",
    "Last_Includes all lodging merchants",
    "Last_Mail Order / Telephone Order Providers",
    "Last_Miscellaneous Stores",
    "Last_Others",
    "Last_Professional Services and Membership Organizations",
    "Last_Repair Services",
    "Last_Retail Stores",
    "Last_Service Providers",
    "Last_Transportation",
    "Last_Utilities",
    "Last_Wholesale Distributors and Manufacturers",
]
_konto_profile_cols = [
    "YearMonth",
    "PersonId",
    "AccountId",
    "ProductId",
    "DistributorId",
    "AgeGroup2",
    "Gender",
    "kommunenavn",
    "MonthsSinceAccountCreatedNum",
    "CreditLimitAmt",
]


def konto_behavioural_data(konto_frame):
    behave_df = konto_frame[_konto_behave_cols]
    return behave_df


def konto_transactional_data(konto_frame):
    trans_df = konto_frame[_konto_trans_cols]
    return trans_df


def konto_profile_data(konto_frame):
    profile_df = konto_frame[_konto_profile_cols]
    df_sorted = profile_df.sort_values(["PersonId", "AccountId", "YearMonth"])
    df_last = df_sorted.drop_duplicates(subset=["PersonId", "AccountId"], keep="last")
    return df_last


# Relativt treg på grunn av sekvensiell for-løkke søk, men har foreløpig ingen bedre måter å sikre at det blir riktig på. Må huske å tenke på at en person kan ha flere accounts når man analyserer
def add_relevant_collectionid_to_konto_frame(
    konto_frame: pd.DataFrame, collection_frame: pd.DataFrame
) -> pd.DataFrame:
    col_copy = collection_frame.copy()
    kont_copy = konto_frame.copy()
    collection_id_dates = (
        col_copy[
            [
                "Collectionid",
                "PersonId",
                "Registreringsmaaned",
                "CollectionOpenedDate",
                "CollectionClosedDate",
            ]
        ]
        .groupby("Collectionid")
        .tail(1)
    )
    kont_copy["Collectionid"] = pd.NA
    kont_copy["Collectionid"] = kont_copy["Collectionid"].astype("object")

    for _, row1 in collection_id_dates.iterrows():
        mask = (
            (kont_copy["PersonId"] == row1["PersonId"])
            & (
                (kont_copy["YearMonth"] < row1["CollectionClosedDate"])
                | pd.isna(row1["CollectionClosedDate"])
            )
            & (kont_copy["Collectionid"].isna())
        )
        kont_copy.loc[mask, "Collectionid"] = row1["Collectionid"]

    return kont_copy


def get_sequences_side_by_side_id_based(df, ids, column):
    sequences = {}
    for i, pid in enumerate(ids, start=1):
        seq = df[df["PersonId"] == pid][column].reset_index(drop=True)
        sequences[f"{pid}"] = seq

    return pd.DataFrame(sequences)


def get_sequences_side_by_side_id_and_columns(df, ids, columns):
    sequences = {}

    for col in columns:
        for pid in ids:
            seq = df[df["PersonId"] == pid][col].reset_index(drop=True)
            sequences[f"{pid}_{col}"] = seq

    return pd.DataFrame(sequences)

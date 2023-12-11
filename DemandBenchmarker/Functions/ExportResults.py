def export_to_excel(dataframes, filepath):
    with pd.ExcelWriter(filepath) as writer:
        for i, df in enumerate(dataframes):
            df.to_excel(writer, sheet_name=f"Sheet{i+1}", index=False)

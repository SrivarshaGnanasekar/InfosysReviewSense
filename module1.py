import pandas as pd
import re
import string

STOPWORDS=["is","the","and","a","an","to","of","in","on","for","with","this","that","it",
           "was","are","as","at","be","by","from","or","but"
           ]

def clean_text(text):
    text=str(text).lower()
    text=re.sub(r"https\S+","",text)
    text=re.sub(r"\d+","",text)
    text=text.translate(str.maketrans("","",string.punctuation))
    words=[W for W in text.split() if W not in STOPWORDS]
    return " ".join(words)

def main():
    file_path_excel="ReviewSense_Customer_Feedback_5000.xlsx"
    file_path_csv="Module1_Cleaned_Feedback.csv"
    
    try:
        df=pd.read_excel(file_path_excel)
        print("Reading from Excel file")
    except FileNotFoundError:
        print("Excel file not found. Attempting to read from existing CSV")
        try:
            df=pd.read_csv(file_path_csv)
            if "cleaned_feedback" in df.columns:
                print("CSV already has cleaned feedback. Skipping Cleaning.")
                return
            elif "feedback" in df.columns:
                print("Re-cleaning feedback from CSV.")
            else:
                raise ValueError("'feedback' column not found in CSV file")
        except FileNotFoundError:
            raise ValueError(" Neither Excel nor CSV file found. Please provide the input file.")
    if "feedback" not in df.columns:
        raise ValueError("'feedback' column not found in file")
    df["clean_feedback"]=df["feedback"].apply(clean_text)
    df.to_csv("Module1_Cleaned_Feedback.csv",index=False)
    print("Module 1 completed successfully")
    print(df[["feedback","clean_feedback"]].head())

if __name__=="__main__":
    main()
    
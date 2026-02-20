import pandas as pd
from collections import Counter
import re

#----   Keyword Cleaning  ----#
def extract_keywords(text):
    text=str(text).lower()
    text=re.sub(r"[^a-z\s]","",text)
    words= text.split()
    return words

#-----  Main Execution  ----#
if __name__=="__main__":
    
    #input from milestone 2
    df=pd.read_csv("Module2_Sentiment_Results_new.csv")
    
    all_words=[]
    df["clean_feedback"].apply(lambda x: all_words.extend(extract_keywords(x)))
    
    #---- Count keyword frequency  ----#
    keyword_freq=Counter(all_words)
    
    #---- Convert to Dataframe  ----#
    keywords_df=pd.DataFrame(keyword_freq.items(),columns=["keyword","frequency"]).sort_values(by="frequency", ascending=False)
    
    #---- Save Results  ----#
    keywords_df.to_csv("Module3_Keyword_Insights.csv",index=False)
    
    print("Module 3 Completed Successfully!")
    
    print(keywords_df.head(10))
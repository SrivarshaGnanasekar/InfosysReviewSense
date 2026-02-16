from textblob import TextBlob
text=input("Enter some text:")
score=TextBlob(text).sentiment.polarity

print("positive" if score>0 else "negative " if score<0 else "neutral")
print(f"({score:.2f})")

import json
import requests
import numpy as np
from pprint import pprint
from openai import OpenAI
from fpdf import FPDF
from dotenv import load_dotenv
import os


load_dotenv()


# Sentiment Analysis Report Generator
COMPANY = "AMZN"
VANTAGE_APIKEY = os.getenv("VANTAGE_APIKEY")
OPENAI_APIKEY = os.getenv("OPENAI_APIKEY")
client = OpenAI(api_key=OPENAI_APIKEY)

def main():
    real = input("Real Data? (y/n) \n")
    if real.lower() == 'n':
        article = skeletonResponse()
        createFakeArticle(article)
        print("Sameple Report Created.")
    else:
        data = getData()
        processedData = processData(data)
        article = AIResponse(processedData)
        createArticle(article)
        print("Report Created.")

# Get sentiment data on a certain company
def getData() -> json:
    dataURL= f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={COMPANY}&apikey={VANTAGE_APIKEY}'
    try:
        data = requests.get(dataURL)
        data = data.json()
    except:
        print("Error Getting Data.")
    return data

# Process Data
def processData(data) -> dict:
    sentimentScore = getSentimentScores(data)
    sentimentArticles = getSentimentArticles(data)
    
    return {
        "AverageSentimentScore": float(sentimentScore),
        "SentimentLabel": getSentimentLabel(sentimentScore),
        "Articles": sentimentArticles
    }
    
    

# Get sentiment scores
def getSentimentScores(data) -> float:
    sentimentScores = []
    for article in data["feed"]:
        tickerSentiment = article["ticker_sentiment"]
        companySentiment = next((item for item in tickerSentiment if item["ticker"] == COMPANY), None)
        sentimentScores.append(float(companySentiment['ticker_sentiment_score']))
        
    averageScore = np.mean(sentimentScores)    
    return averageScore


# Get most relevant articles
def getSentimentArticles(data) -> dict:
    sentimentArticles = []
    for article in data["feed"]:
        articleSummary = {}
        articleSummary["URL"] = article["url"]
        articleSummary["Date"] = article["time_published"]
        company_sentiment = next((item for item in article['ticker_sentiment'] if item["ticker"] == COMPANY), None)
        articleSummary['Relevance'] = float(company_sentiment['relevance_score'])
        sentimentArticles.append(articleSummary)
    
    sentimentArticles = sorted( sentimentArticles, key=lambda x: x['Relevance'], reverse=True )
    
    return sentimentArticles[:5]

# Get sentiment label
def getSentimentLabel(score):
    if score <= -0.35:
        return "Bearish"
    elif -0.35 < score <= -0.15:
        return "Somewhat-Bearish"
    elif -0.15 < score < 0.15:
        return "Neutral"
    elif 0.15 <= score < 0.35:
        return "Somewhat-Bullish"
    elif score >= 0.35:
        return "Bullish"


# Configure Pete
def AIResponse(processedData) -> None:
    prompt = createPrompt(processedData)
    systemPrompt = createSystemPrompt()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": systemPrompt},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content
    
def skeletonResponse():
    return """Apple Inc. (AAPL) currently has a sentiment score of 0.1171, which is labeled as Neutral. This score is derived from averaging fifty different articles, indicating a mixed perception among investors and analysts. The articles suggest relevant issues surrounding potential changes in the company's credit card partner, fluctuations in iPhone sales in China, stock projections, consumer product testimonies, and overall investor concerns about the company's direction and challenges.

        The selection of articles provides an insightful snapshot into Apple's current landscape. One significant article highlights Apple's potential move away from Goldman Sachs, its current credit card partner, to either Barclays or Synchrony. This strategic shift indicates a possible restructuring in Apple's financial services approach. Another key article points to declining iPhone sales in China, a market of critical importance to Apple's revenue streams. This trend might reflect broader economic challenges or competitive pressures in this significant market. Additionally, an article about Apple's longer-term stock potential posits varying perspectives, with analysts debating its performance in the coming years, emphasizing Apple's pivotal role in innovation and market adaptability. On the tech innovation front, the Apple Watch's ability to save a life by providing emergency alerts showcases Apple's successful entry into health monitoring, underlining consumer trust and product relevance. Conversely, some reports illustrate anxiety about Apple's future amidst several competitive and operational hurdles.

        In conclusion, despite experiencing mixed sentiment and market challenges, Apple maintains a steady, neutral outlook. Its strategic endeavors and innovation across product segments continue to underpin its significant market position. Investors remain watchful, analyzing current challenges against long-term viability and growth potential."""

def createSystemPrompt() -> str:
    prompt = (
        "You are a financial analyst. You are to create a report based on data supplied by the user."
        "You do not need a title."
        "Do not use any markdown. Only Pure text"
        "You must only use characters compatible with latin-1 encoding, avoiding special characters like curly apostrophes, quotes, or non-ASCII symbols. "
        "You are to create a short summary of the data entered by the user (~100 words). "
        "You are then to analyze and get the key insights and themes from a set of articles provided. "
        "You includes dates when possible when referencing articles"
        "This is the main body and should be around 200 words. "
        "You need to finish with a conclusion of only ~50 words. Highlight the main takeaway on the current state of the company."
    )
    return prompt

# Create Prompt
def createPrompt(processedData) -> str:
    roundedScore = round(processedData['AverageSentimentScore'], 3)
    prompt = (
        f"The sentiment score for the company given by the ticker {COMPANY} is {roundedScore}. "
        f"This sentiment score was calculated by calculating the average over 50 articles. "
        f"This sentiment score has the equivalent label: {processedData['SentimentLabel']}. "
        f"The most relevant articles are in the following json file: {processedData['Articles']}. "
        f"They have a Relevance score, URL, and Date."
    )
    return prompt

def createArticle(article):
    pdf_filename = "AI_Financial_Impact.pdf"
    processedData = processData(getData()) 
    createPDF(article, pdf_filename, processedData['AverageSentimentScore'], processedData['SentimentLabel'])
    
def createFakeArticle(article):
    pdf_filename = "AI_Financial_Impact.pdf"
    createPDF(article, pdf_filename, 0.120123, "Neutral")

    
def createPDF(article_text, pdf_name, sentiment_score, sentiment_label):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=10)
    pdf.add_page()

    pdf.set_font("Times", style="B", size=16)
    pdf.cell(0, 10, f"Sentiment Analysis Summary For {COMPANY}", ln=True, align="C")
    pdf.ln(5)

    bar_width = 100
    bar_x = 55
    bar_y = 30
    pdf.set_draw_color(0, 0, 0)

    if sentiment_label == "Neutral":
        pdf.set_fill_color(0, 0, 139)  
    elif sentiment_label in ["Bearish", "Somewhat-Bearish"]:
        pdf.set_fill_color(139, 0, 0)  
    elif sentiment_label in ["Bullish", "Somewhat-Bullish"]:
        pdf.set_fill_color(0, 100, 0)  



    normalized_score = (sentiment_score + 1) / 2 
    filled_width = normalized_score * bar_width

    pdf.rect(bar_x, bar_y, bar_width, 10)  
    pdf.rect(bar_x, bar_y, filled_width, 10, 'F') 

    pdf.set_y(bar_y + 15)
    pdf.set_font("Times", style="IB", size=14)  
    pdf.cell(0, 15, f"Sentiment Score: {sentiment_score:.2f} ({sentiment_label})", ln=True, align="C")
    pdf.set_font("Times", size=12)  

    pdf.set_font("Times", size=12)
    for line in article_text.split("\n"):
        pdf.multi_cell(0, 9, line)

    pdf.output(pdf_name)

    
main()
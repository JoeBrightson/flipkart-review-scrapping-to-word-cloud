import time
import pandas as pd
import nltk
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from selenium import webdriver
from wordcloud import WordCloud, STOPWORDS
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from nltk.sentiment import SentimentIntensityAnalyzer

# ✅ Fix for VS Code interactive plotting
matplotlib.use("TkAgg")  # Enables interactive mode

# Download VADER lexicon
nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

# Configure Selenium WebDriver
chrome_options = Options()
chrome_options.add_argument("--headless")  # Run without UI
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_options)

# Flipkart Product Review URL
url = "https://www.flipkart.com/casio-a-158wa-1df-vintage-a158wa-1df-black-dial-silver-stainless-steel-band-digital-watch-men-women/product-reviews/itmf3zhdga85ghju?pid=WATDJ5YXGFUP5RNG&lid=LSTWATDJ5YXGFUP5RNG3Q3EEO&marketplace=FLIPKART"
driver.get(url)
time.sleep(5)

# ✅ Auto-detect Flipkart review class (HTML structure may change)
possible_classes = ["t-ZTKy", "ZmyHeo"]  # Common Flipkart review classes
review_class = None

for cls in possible_classes:
    try:
        WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.CLASS_NAME, cls)))
        review_class = cls
        print(f"Using detected review class: {review_class}")
        break
    except Exception:
        continue

if review_class is None:
    print("Failed to detect the review class. Flipkart might have changed the website structure.")
    driver.quit()
    exit()

all_reviews = set()
previous_count = 0  # Track review count before pagination

while len(all_reviews) < 100:
    WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.CLASS_NAME, review_class)))
    reviews = driver.find_elements(By.CLASS_NAME, review_class)

    for review in reviews:
        text = review.text.strip()
        if text:
            all_reviews.add(text)
        if len(all_reviews) >= 100:
            break

    print(f"Collected {len(all_reviews)} reviews so far...")

    # Stop if no new reviews are collected
    if len(all_reviews) == previous_count:
        print("No new reviews found. Ending collection.")
        break
    previous_count = len(all_reviews)

    # Try to click the Next button
    try:
        next_button = driver.find_element(By.XPATH, "//span[text()='Next']")
        driver.execute_script("arguments[0].click();", next_button)
        time.sleep(3)
    except Exception:
        print("No more pages available. Stopping...")
        break

driver.quit()

# Store reviews in a DataFrame
df = pd.DataFrame(list(all_reviews), columns=["Review"])
if df.empty:
    print("No reviews found! The website structure may have changed.")
else:
    print(f"\nTotal {len(df)} Reviews Scraped Successfully!")

# Perform Sentiment Analysis
df["Sentiment_Score"] = df["Review"].apply(lambda review: sia.polarity_scores(review)["compound"])
df["Sentiment"] = df["Sentiment_Score"].apply(lambda score: "Positive" if score > 0 else "Negative")

df.to_csv("flipkart_reviews_with_sentiment.csv", index=False, encoding="utf-8")
print("\nSentiment analysis saved to flipkart_reviews_with_sentiment.csv")
positive_reviews = " ".join(df[df["Sentiment"] == "Positive"]["Review"])
negative_reviews = " ".join(df[df["Sentiment"] == "Negative"]["Review"])

stopwords = set(STOPWORDS)
stopwords.update(["flipkart", "product", "watch", "buy", "good", "bad"])  
if positive_reviews:
    wordcloud_positive = WordCloud(width=800, height=400, background_color="white", colormap="Greens", stopwords=stopwords).generate(positive_reviews)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud_positive, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud - Positive Reviews")
    plt.savefig("positive_wordcloud.png")
    plt.show()
else:
    print("No positive reviews found! Word Cloud cannot be generated.")


if negative_reviews:
    wordcloud_negative = WordCloud(width=800, height=400, background_color="white", colormap="Reds", stopwords=stopwords).generate(negative_reviews)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud_negative, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud - Negative Reviews")
    plt.savefig("negative_wordcloud.png")  
    plt.show()
else:
    print("No negative reviews found! Word Cloud cannot be generated.")


if not df["Sentiment"].empty:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="Sentiment", palette={"Positive": "green", "Negative": "red"})
    plt.title("Sentiment Analysis of Flipkart Reviews")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.savefig("sentiment_analysis.png")
    plt.show()
else:
    print("No sentiment data available for visualization.")
 
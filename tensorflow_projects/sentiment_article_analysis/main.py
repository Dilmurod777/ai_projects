from textblob import TextBlob
from newspaper import Article

urls = [
    'https://en.wikipedia.org/wiki/Mathematics',
    'https://edition.cnn.com/2022/12/21/asia/the-serpent-charles-sobhraj-nepal-release-intl/index.html',
    'https://edition.cnn.com/2022/12/21/football/shaun-botterill-photographer-lionel-messi-instagram-most-liked-spt-intl/index.html',
    'https://edition.cnn.com/style/article/kim-mi-soo-dead-intl-scli/index.html?dicbo=v2-44682ba5ee0b90166f81d79e5cd13eeb',
    'https://edition.cnn.com/2022/03/31/world/frozen-zoo-save-species-scn-c2e-spc-intl/index.html',
]

analysis_outputs = {
    'Very Negative': [-1, -0.5],
    'Negative': [-0.5, -0.05],
    'Neutral': [-0.05, 0.05],
    'Positive': [0.05, 0.5],
    'Very Positive': [0.5, 1]
}

for url in urls:
    article = Article(url)

    article.download()
    article.parse()
    article.nlp()

    text = article.text
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity

    for analysis in analysis_outputs:
        if analysis_outputs[analysis][0] <= sentiment <= analysis_outputs[analysis][1]:
            print(f"{article.title} - {analysis}")

class Tweet:

    def __init__(self, text, date, author_distance, author_content, location, language, url, image, num_retweets):
        if date == "#NV" or date == "N/" : date = 0
        if author_distance == "#NV" or author_distance == "N/" : author_distance = 0
        if location == "#NV" or location == "N/" : location = 0
        if language == "#NV" or language == "N/" : language = 0
        if url == "#NV" or url == "N/" : url = 0
        if image == "#NV" or image == "N/" : image = 0
        if num_retweets == "#NV" or num_retweets == "N/" : num_retweets = 0
        self.text = text
        self.date = date
        self.author_distance = int(author_distance)
        self.author_content = author_content
        self.location = location
        self.language = language
        self.url = int(url)
        self.image = int(image)
        self.num_retweets = int(num_retweets)

    def set_embedding_vector(self, embedding):
        self.embedding = embedding

    def set_cluster_label(self, label):
        self.label = label

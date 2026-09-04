import time
import random
import uuid
import yaml
from datetime import datetime
from database_manager import DBManager

# Highly variable templates to generate realistic tweets and comments
TEMPLATES = {
    "Apple": {
        "Positive": [
            "Just got the new iPhone! The camera zoom is absolutely mind-blowing. 📸😍",
            "Apple customer service replaced my cracked screen in 30 minutes. Amazing experience! 🙌",
            "The new MacBook Pro battery life is insane. Easily getting 15+ hours of heavy dev work.",
            "Honestly, Apple AirPods Pro are worth every single penny. Best noise cancellation out there. 🎧",
            "iOS update feels incredibly smooth and snappy. Love the new widget customizations!"
        ],
        "Negative": [
            "Why does Apple keep raising prices? $1200 for a phone is getting ridiculous. 😡",
            "My iPhone battery is draining so fast after the recent software update. Fix this please @Apple!",
            "So annoyed that Apple removed the charger from the box. Peak corporate greed. 🔌❌",
            "MacBook keyboard is sticking again. Extremely disappointed with this design. 🙄",
            "Spent 2 hours waiting at the Apple store Genius Bar. Service is terrible lately."
        ],
        "Neutral": [
            "Anyone know if the new iPad supports the older Apple Pencil?",
            "Comparing Apple Watch Series 9 specs with the Ultra. Deciding which to get.",
            "Apple is announcing their quarterly earnings report later this afternoon.",
            "Just read an article about Apple's upcoming supply chain updates.",
            "Does the Apple Store in New York require appointments for repairs?"
        ]
    },
    "Netflix": {
        "Positive": [
            "This new documentary series on Netflix is a masterpiece! Must watch. 🎬🍿",
            "Finished binging the new season of Stranger Things. Absolutely incredible ending! 🤯",
            "Netflix is hitting it out of the park with their international shows lately.",
            "Love that Netflix released all episodes at once. Long live the binge watch! 📺",
            "The cinematography in the new Netflix original film is stunning."
        ],
        "Negative": [
            "Netflix cancels another amazing show after just one season. I'm done. Unsubscribing! 🤬",
            "Why is Netflix raising subscription prices while cracking down on password sharing? 📉",
            "The Netflix app keeps crashing on my smart TV. Works fine on everything else though.",
            "Netflix's library has been feeling really dry lately. Hard to find anything worth watching.",
            "The streaming quality is buffering constantly tonight. Netflix servers must be struggling."
        ],
        "Neutral": [
            "What's new on Netflix this weekend? Send recommendations.",
            "Netflix is releasing a new thriller movie next Friday.",
            "Does anyone else share a Netflix plan with their family?",
            "Netflix stock price fluctuated after the latest subscriber statistics report.",
            "Searching for a good comedy show to watch on Netflix tonight."
        ]
    },
    "Starbucks": {
        "Positive": [
            "Starting my Monday morning with a fresh iced caramel macchiato. Life is good! ☕✨",
            "The barista at my local Starbucks is always so friendly and makes my day. 😊",
            "Obsessed with the new seasonal pumpkin spice cream cold brew!",
            "Starbucks mobile ordering is a lifesaver. Walked in and picked it up immediately. 🏃‍♂️💨",
            "Got a free drink today using my Starbucks rewards stars! Loving the loyalty program."
        ],
        "Negative": [
            "Starbucks got my coffee order completely wrong again. Ordered oat milk, got whole milk. 🥛🤦‍♂️",
            "Why does a medium coffee at Starbucks cost almost $7 now? Insane inflation.",
            "Waiting in the Starbucks drive-thru for 25 minutes. Worst service ever. 😤",
            "They burnt the espresso shots. My coffee tastes like pure charcoal today.",
            "Local Starbucks has zero seating available. Everything is blocked off. Pointless."
        ],
        "Neutral": [
            "Is there a Starbucks nearby that has a drive-thru?",
            "What time does Starbucks open on weekends?",
            "Checking if the Starbucks holiday cups are available yet.",
            "Does Starbucks offer free WiFi at all of their cafes?",
            "Starbucks has launched a new line of reusable tumblers."
        ]
    },
    "Google": {
        "Positive": [
            "The camera on the Google Pixel is miles ahead of any other smartphone. 📸✨",
            "Google Docs' voice typing is incredibly accurate. Saved me hours of writing today.",
            "Google Maps offline download feature saved me during my road trip through the mountains. 🗺️🙌",
            "Loving the new search shortcuts Google added. Makes navigation so much faster.",
            "Google Translate is seriously like magic. Translating real-time audio instantly."
        ],
        "Negative": [
            "Google search results are becoming so cluttered with ads and sponsored content. 🙄",
            "Chrome browser is eating up 90% of my computer RAM. Laptop feels like it's melting! 💻🔥",
            "Google Drive keeps failing to sync my work files today. Super frustrating.",
            "My Pixel phone is overheating constantly when charging. Google support was no help.",
            "Sad to see Google killing off another useful service. The Google graveyard is huge."
        ],
        "Neutral": [
            "Google is updating its search algorithms again this month.",
            "How do I clear my search history on Google Chrome?",
            "Google Cloud platform is hosting a developer summit next Tuesday.",
            "Comparing Google Workspace pricing with Microsoft Office 365.",
            "Google's parent company Alphabet announced their new AI model release."
        ]
    }
}

PLATFORMS = ["Twitter", "YouTube Comments", "Reddit"]
USERNAMES = [
    "tech_guru", "coffee_lover", "binge_watcher", "data_nerd", "code_monkey",
    "wanderlust", "pixel_fan", "critic_pro", "daily_rant", "gamer_girl",
    "market_watcher", "news_feed", "silicon_valley", "foodie_gram", "chill_vibes"
]

class DataGenerator:
    def __init__(self, config_path="config/config.yaml"):
        # Load configuration
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
            
        self.db = DBManager(config_path)
        self.brands = self.config["simulation"]["brands"]
        # Ensure we only use brands for which we have templates
        self.active_brands = [b for b in self.brands if b in TEMPLATES]
        if not self.active_brands:
            self.active_brands = list(TEMPLATES.keys())
            
    def generate_single_post(self):
        """Generates a realistic social media post dictionary."""
        brand = random.choice(self.active_brands)
        sentiment = random.choice(["Positive", "Negative", "Neutral"])
        text = random.choice(TEMPLATES[brand][sentiment])
        platform = random.choice(PLATFORMS)
        username = random.choice(USERNAMES) + str(random.randint(10, 99))
        
        # Follower scale depending on user type
        followers = random.randint(50, 1500)
        if random.random() < 0.15: # 15% chance of influencer/high followers
            followers = random.randint(10000, 250000)
            
        # Engagement variables proportional to followers
        likes = int(followers * random.uniform(0.01, 0.08)) + random.randint(0, 5)
        retweets = int(likes * random.uniform(0.1, 0.4)) if platform == "Twitter" else 0
        
        return {
            "tweet_id": str(uuid.uuid4()),
            "platform": platform,
            "username": username,
            "followers": followers,
            "timestamp": datetime.now().isoformat(),
            "text": text,
            "topic": brand,
            "likes": likes,
            "retweets": retweets
        }

    def start_simulation(self, delay=1.5, batch_size=2):
        """Infinite loop inserting simulated social posts into SQLite."""
        self.db.initialize_db()
        print(f"Simulation started. Delay: {delay}s, Batch size: {batch_size}. Press Ctrl+C to stop.")
        
        try:
            while True:
                posts = [self.generate_single_post() for _ in range(batch_size)]
                inserted = self.db.insert_raw_posts(posts)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Simulated stream: {inserted} posts written to SQLite database.")
                time.sleep(delay)
        except KeyboardInterrupt:
            print("\nSimulation stopped by user.")

if __name__ == "__main__":
    generator = DataGenerator()
    # Read delay from config if exists
    delay = generator.config["simulation"].get("delay_seconds", 1.5)
    generator.start_simulation(delay=delay, batch_size=2)

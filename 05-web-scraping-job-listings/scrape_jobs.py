#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Job Listings Web Scraper
Author: Lavanya Siripuram

GOAL
----
Scrape structured job listing data (title, company, location, date posted)
from a public practice site, then do a small amount of analysis on the
result — turning "I can extract HTML" into "I can extract HTML and produce
something useful from it."

SITE
----
https://realpython.github.io/fake-jobs/ — a static site built by the
Real Python team specifically for web scraping practice. All listings are
fictional ("fake jobs"), and the site is intentionally public and
scraping-friendly, so there are no robots.txt/ethics concerns in using it
for this kind of exercise.

NOTE ON RUNNING THIS
---------------------
This script makes a live HTTP request, so it needs an internet connection
and won't run inside network-restricted sandboxes. It runs cleanly on a
normal machine with internet access.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt

URL = "https://realpython.github.io/fake-jobs/"

# ---------------------------------------------------------------------------
# 1. Fetch the page
# ---------------------------------------------------------------------------
response = requests.get(URL, timeout=10)
response.raise_for_status()  # fail loudly if the request didn't succeed
print(f"Fetched {URL} — status {response.status_code}, "
      f"{len(response.text):,} characters of HTML")

# ---------------------------------------------------------------------------
# 2. Parse and locate the job cards
# ---------------------------------------------------------------------------
soup = BeautifulSoup(response.content, "html.parser")
results = soup.find(id="ResultsContainer")
job_cards = results.find_all("div", class_="card-content")
print(f"Found {len(job_cards)} job listings\n")

# ---------------------------------------------------------------------------
# 3. Extract structured fields from each card
# ---------------------------------------------------------------------------
jobs = []
for card in job_cards:
    title = card.find("h2", class_="title")
    company = card.find("h3", class_="company")
    location = card.find("p", class_="location")
    date_posted = card.find("time")

    jobs.append({
        "title": title.text.strip() if title else None,
        "company": company.text.strip() if company else None,
        "location": location.text.strip() if location else None,
        "date_posted": date_posted.text.strip() if date_posted else None,
    })

df = pd.DataFrame(jobs)

# ---------------------------------------------------------------------------
# 4. Save the structured result
# ---------------------------------------------------------------------------
df.to_csv("job_listings.csv", index=False)
print(f"Saved {len(df)} listings to job_listings.csv\n")
print(df.head())

# ---------------------------------------------------------------------------
# 5. A little analysis on top of the scraped data
# ---------------------------------------------------------------------------
# Extract the region code from location (e.g. "Stewartbury, AA" -> "AA").
# This dataset uses fictional 2-letter region codes rather than real
# country/state codes, so this is a structural pattern in the test data,
# not a real geography.
df["region"] = df["location"].str.split(", ").str[-1]

print("\n--- Listings by region ---")
region_counts = df["region"].value_counts()
print(region_counts)

# Which companies have more than one listing?
print("\n--- Companies with multiple listings ---")
company_counts = df["company"].value_counts()
print(company_counts[company_counts > 1])

# How many listings mention "Python" in the title? (this site is themed
# around Python jobs, so this is a natural thing to check)
python_titles = df[df["title"].str.contains("Python", case=False, na=False)]
print(f"\n{len(python_titles)} of {len(df)} listings mention 'Python' in the title:")
print(python_titles["title"].tolist())

# ---------------------------------------------------------------------------
# 6. Plot: listings by region
# ---------------------------------------------------------------------------
plt.figure(figsize=(7, 5))
region_counts.plot(kind="bar", color="#2E5C8A")
plt.title("Job Listings by Region Code")
plt.xlabel("Region")
plt.ylabel("Number of Listings")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("listings_by_region.png", dpi=150)
print("\nSaved plot to listings_by_region.png")

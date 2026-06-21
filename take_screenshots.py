import time
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1200, "height": 800})
    
    # Landing page
    page.goto('http://127.0.0.1:5000/')
    time.sleep(2)
    page.screenshot(path='screenshots/landing.png')
    
    # Result page
    page.goto('http://127.0.0.1:5000/checkpage')
    time.sleep(1)
    
    email_text = """Subject: You won! Claim your Amazon gift card now

Dear Customer,

Congratulations! You have been selected as one of our lucky winners. Click the link below to verify your identity and claim your FREE $500 Amazon Gift Card today!

http://amaz0n-rewards-verify.com/claim

This offer expires in 24 hours.

Best regards,
Amazon Rewards Team"""

    page.fill('textarea[name="email_text"]', email_text)
    page.click('button[type="submit"]')
    time.sleep(2)
    page.screenshot(path='screenshots/result.png')
    
    browser.close()

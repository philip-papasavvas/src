"""
Created on: 10 Oct 2019
Author: Philip.P

Script to send automated email of the daily WOD and an
inspirational quote to the user
"""

# STEPS
# done - 1 - Make email client send email from user account
# done - 2- Use requests/beautiful soup to scrape online to get the WODs
# done - 3- Get a large bank of inspirational quotes
# done - 4 - Send email
# 5 - Store down the data in a database

# import smtplib, ssl
#
# port = 465  # For SSL
# smtp_server = "smtp.gmail.com"
# sender_email = "pythonpaps@gmail.com"  # Enter your address
# receiver_email = "philip.papasavvas@gmail.com"  # Enter receiver address
# password = input("Type your password and press enter: ")
# message = """\
# Subject: Hi there
#
# This message is sent from Python."""
#
# context = ssl.create_default_context()
# with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
#     server.login(sender_email, password)
#     server.sendmail(sender_email, receiver_email, message)

# 1. EMAIL PART
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import getpass

sender_email = "pythonpaps@gmail.com"
receiver_email = "philip.papasavvas@gmail.com"
password = getpass.getpass()

message = MIMEMultipart("alternative")
message["Subject"] = "CrossFit Open 2020"
message["From"] = sender_email
message["To"] = receiver_email

# Create the plain-text and HTML version of your message
text = """\
Hi,
CrossFit Open 20.1 starts very soon, see more at https://www.crossfit.com/
"""
html = """\
<html>
  <body>
    <p>Hi,<br>
       CrossFit Open 20.1 starts very soon.<br>
       Head to <a href="https://www.crossfit.com/">CrossFit</a> 
       to see more!
    </p>
  </body>
</html>
"""

# Turn these into plain/html MIMEText objects
part1 = MIMEText(text, "plain")
part2 = MIMEText(html, "html")

# Add HTML/plain-text parts to MIMEMultipart message
# The email client will try to render the last part first
message.attach(part1)
message.attach(part2)

# Create secure connection with server and send email
context = ssl.create_default_context()
with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
    server.login(sender_email, password)
    server.sendmail(
        sender_email, receiver_email, message.as_string()
    )


# 2. WEB SCRAPING TO GET THE DATA
import requests
from bs4 import BeautifulSoup

website = 'http://www.crossfit.com'
page = requests.get(website)

# Create a BeautifulSoup object
soup = BeautifulSoup(page.text, 'html.parser')

# the below is defined for the part where we see the Workout of the Day
repo = soup.find(class_="sf1KM0yG_abhAm_QFxP7E")

#below is the specific bit we want for the actual workout and where it is contained
repo_list = repo.find_all(class_="_6zX5t4v71r1EQ1b1O0nO2 LerElceKJ8FwMmx8LCILf")

# get the different lines of the workout
for repo in repo_list:
    # find the first <p> tag and get the text. Split the text using splitlines() as the website will do
    # this for the workouts
    repo_contents = repo.find_all('p').text.splitlines()


# 3. PULL RANDOM INSPIRATIONAL QUOTES FROM THE DATABASE
import pymongo
import random

# insert my password here
client = pymongo.MongoClient("mongodb+srv://master_user:<password>@cluster0-iooan.mongodb.net/test?retryWrites=true&w=majority")
library = client['email'] # type pymongo.database.Database
library.list_collection_names() # lists collection names
type(library['quotes']) # type pymongo.collection.Collection
coll = library['quotes'] # collection
quote_dict = coll.find_one() # dict of quotes
random_quote_key = random.choice(list(quote_dict.keys()))
random_quote = quote_dict[random_quote_key]

# 4- SEND EMAIL
import datetime

date_today_dt = datetime.datetime.now()
date_today_str = date_today_dt.strftime("%Y%m%d")

sender_email = "pythonpaps@gmail.com"
receiver_email = "philip.papasavvas@gmail.com"
password = getpass.getpass()

message = MIMEMultipart("alternative")
message["Subject"] = " ".join([date_today_str, "CrossFit Daily WOD Email"])
message["From"] = sender_email
message["To"] = receiver_email

msg_body = "WOD for {date}: \n {workout}. \n \n {quote} - {person}".format(date=date_today_str,
                                                                                 workout=repo_contents[0],
                                                                                 quote = random_quote,
                                                                                 person= random_quote_key)
msg_body_html = "WOD for <strong>{date}</strong>:" \
                "<p>" \
                " <strong> {workout} </strong>" \
                "</p>" \
                "<br> <br> <strong> {quote} </strong> " \
                "<em> {person} </em>".format(date=date_today_str, workout=repo_contents[0],
                                               quote = random_quote, person= random_quote_key)

# Turn these into plain/html MIMEText objects
part1 = MIMEText(msg_body, "plain")
part2 = MIMEText(msg_body_html, "html")

# Add HTML/plain-text parts to MIMEMultipart message
# The email client will try to render the last part first
message.attach(part1)
message.attach(part2)

# Create secure connection with server and send email
context = ssl.create_default_context()
with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
    server.login(sender_email, password)
    server.sendmail(
        sender_email, receiver_email, message.as_string()
    )
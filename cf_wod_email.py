"""
Created on: 10 Oct 2019
Author: Philip.P

Script to send automated email of the daily WOD and an
inspirational quote to the user.

"""
import os
import json
import re
import random
import datetime
from utils import get_config_path, get_path, get_db_path, get_import_path

import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import getpass

import requests
from bs4 import BeautifulSoup

import pymongo

def update_quote_db(mongo_config, quotes_import_json):
    """Function to update database with quotes, mongo_config and quote_import_json as inputs"""

    user = mongo_config['mongo_user']
    password = mongo_config['mongo_pwd']
    mongo_url = mongo_config['url_cluster']

    host_url = "".join(["mongodb+srv://", user, ":", password, "@", mongo_url])
    client = pymongo.MongoClient(host_url)

    library = client['email']  # type pymongo.database.Database
    coll = library['quotes']  # collection
    existing_db = coll.find_one()

    quotes_to_add = dict(quotes_import_json)

    coll.update(existing_db, quotes_import_json)

    print("Successfully added new quotes to the db")

class CF_email():
    """
    Class to send email of daily WOD and inspirational quote to email
    addresses specified in the config

    Parameters
    ----------
    email_config: json
        json config with the following keys: "subject", "source_email",
        "source_email_pwd", "email_distrib_list"
    mongo_config: json
        json config with the following keys: "mongo_user", "mongo_pwd",
        "url_cluster""
    image: bool, default False
        To print images in the HTML email- CrossFit logo, photo
    """

    def __init__(self, email_config, mongo_config, image=False):

        self.config = {**email_config, **mongo_config}
        self.print_images = image

        date_today_dt = datetime.datetime.now()
        date_today_str = date_today_dt.strftime("%Y%m%d")

        self.date = date_today_str

    def scrape_workout(self, website="http://www.crossfit.com"):
        """
        Function to scrape the Workout Of the Day (WOD) from the CrossFit
        website

        Params
        ------
        website: str, default www.crossfit.com
            Website from which I wish to scrape data

        Returns
        -------
        workout_bs4_element: str
            Workout HTML object parsed by beautiful soup directly from the
            origin website
        """

        page = requests.get(website)
        soup = BeautifulSoup(page.text, 'html.parser')  # Create a BeautifulSoup object

        wod_class = "_6zX5t4v71r1EQ1b1O0nO2 jYZW249J9cFebTPrzuIl0"#"_1txuw01hodoV9pnEgg9vKl _1iFhEb-WHz4xf8bQbBXErH"
        #"_3ISZlXlYytTnehLYeQWJo1" #"sf1KM0yG_abhAm_QFxP7E" # the tag is defined for location of WOD
        repo = soup.find(class_=wod_class)
        # wod_tag = "_6zX5t4v71r1EQ1b1O0nO2 jYZW249J9cFebTPrzuIl0" \
                   # "_6zX5t4v71r1EQ1b1O0nO2 LerElceKJ8FwMmx8LCILf" # tag for where WOD is stored
        workout_bs4_element = repo #repo.find(class_=wod_tag) #beautiful soup element

        if workout_bs4_element is None:
            workout_bs4_element = "The email has broken for some reason, maybe the tag has changed for the WOD. \n" \
                                  "Sorry, this will be fixed soon!"

        return workout_bs4_element

    def get_random_quote(self, user=None, password=None, mongo_url=None):
        """Function to get random quote from database for the email

        Returns
        -------
            author, quote: str, str
        """
        # initialise the database
        print("Retrieving random quote from database")

        if user is None:
            user = self.config['mongo_user']

        if password is None:
            password = self.config['mongo_pwd']

        if mongo_url is None:
            mongo_url = self.config['url_cluster']

        host_url = "".join(["mongodb+srv://",user,":",password,"@",mongo_url])
        client = pymongo.MongoClient(host_url)
        # client = pymongo.MongoClient("mongodb+srv://username:<password>@cluster0-iooan.mongodb.net/test?retryWrites=true&w=majority")

        library = client['email']  # type pymongo.database.Database
        coll = library['quotes']  # collection

        quote_dict = coll.find_one()  # dict of quotes
        quote_dict.pop('_id')

        author = random.choice(list(quote_dict.keys()))
        quote = quote_dict[author]

        print("Quote retrieved!")

        # library.list_collection_names()  # lists collection names
        # type(library['quotes'])  # type pymongo.collection.Collection

        # quote_dict = {author: quote}
        return author, quote

    def generate_email_text(self, wod_text, author, quote):
        """Function to put together message for email"""

        print("Generating email...")
        msg_body_html = "WOD for {date}".format(date=self.date) + "<p>"
        msg_body_html += str(wod_text) + "<p>"
        msg_body_html += "<br> <strong> Quote of the day </strong> </br> <p>"
        msg_body_html += "<em>" + quote + " </em>"
        msg_body_html += "<strong>" + author + "</strong>"

        return msg_body_html

    def send_email(self, body, password=None):
        """Function to send email at the final step to the user
        with the information

        Params
        ------
            body: str
                HTML string
        """
        # password = getpass.getpass()
        if password is None:
            password = self.config['source_email_pwd']

        msg = MIMEMultipart("alternative") # MIMEMultipart(_subtype='related')
        msg["Subject"] = " ".join([self.date, self.config['subject']])
        msg["From"] = self.config['source_email']
        msg["To"] = self.config['email_distrib_list']

        msg_body_html = body
        body = MIMEText(msg_body_html, "html")
        msg.attach(body)

        if self.print_images:
            img_data = open(get_path("crossfit-logo.png"), 'rb').read()
            img = MIMEImage(img_data, 'png')
            img.add_header('Content-Id', '<cf_logo>')  # angle brackets are important
            img.add_header("Content-Disposition", "inline", filename="cf_logo")
            msg.attach(img)

        print("Sending email...")

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", port=465, context=context) as server:
            server.login(msg["From"], password)
            server.sendmail(msg["From"], msg["To"].split(','), msg.as_string())

    def run(self):
        """Run function to bring it all together"""
        wod_text = self.scrape_workout()
        author, quote = self.get_random_quote()

        email_text = self.generate_email_text(wod_text=wod_text, author=author,
                                              quote=quote)

        self.send_email(body = email_text)
        print("Successfully sent {} daily WOD email".format(self.date))

if __name__ == "__main__":

    with open(get_config_path("mongo_private.json")) as mongo_json:
        mongo_config = json.load(mongo_json)

    with open(get_config_path("cf_email_private.json")) as cf_email_json:
        email_config = json.load(cf_email_json)

    rn = CF_email(email_config=email_config, mongo_config=mongo_config, image=True)
    rn.run()

    # with open(get_import_path("new_quotes.json")) as quotes_js:
    #     quotes_json = json.load(quotes_js)
    #
    # update_quote_db(mongo_config=mongo_config, quotes_import_json=quotes_json)
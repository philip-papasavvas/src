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
from utils import get_config_path, get_images_path, get_path, get_db_path, get_import_path

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

    def __init__(self, email_config, mongo_config):

        self.config = {**email_config, **mongo_config}

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


    def access_mongodb(self, user=None, password=None, mongo_url=None):
        """Function to login to MongoDB and initialise library to enable writing"""
        if user is None:
            user = self.config['mongo_user']

        if password is None:
            password = self.config['mongo_pwd']

        if mongo_url is None:
            mongo_url = self.config['url_cluster']

        host_url = "".join(["mongodb+srv://", user, ":", password, "@", mongo_url])
        client = pymongo.MongoClient(host_url)

        return client

    def get_random_quote(self, mongo_client):
        """Function to get random quote from database for the email

        Returns
        -------
            author, quote: str, str
        """
        # initialise the database
        print("Retrieving random quote from database")

        client = mongo_client
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

    def generate_email_text(self, wod_text, get_quote=False, author=None, quote=None):
        """Function to put together message for email"""

        print("Generating email...")
        msg_body_html = "WOD for {date}".format(date=self.date) + "<p>"
        msg_body_html += str(wod_text) + "<p>"
        if get_quote:
            msg_body_html += "<br> <strong> Quote of the day </strong> </br> <p>"
            msg_body_html += "<em>" + quote + " </em>"
            msg_body_html += "<strong>" + author + "</strong>"

        return msg_body_html

    def prepare_mime_format(self, body, embed_image):
        """Method to prepare the MIME format of the email, specifying the HTML body"""

        mime_msg = MIMEMultipart("alternative") # MIMEMultipart(_subtype='related')
        mime_msg["Subject"] = " ".join([self.date, self.config['subject']])
        mime_msg["From"] = self.config['source_email']
        mime_msg["To"] = self.config['email_distrib_list']

        # if embed_image:
        #     msg_body_html = '<br><img src="cid: cf_logo_id"/><br>' + body
        #     img_data = open(get_images_path("crossfit-logo.png"), 'rb').read()
        #     img = MIMEImage(img_data, 'png')
        #     img.add_header('Content-Id', '<cf_logo_id>')  # angle brackets are important
        #
        # msg_body_html = body
        #
        # body = MIMEText(msg_body_html, "html")
        # mime_msg.attach(body)
        #
        # if embed_image:
        #     mime_msg.attach(img)

        # # We reference the image in the IMG SRC attribute by the ID we give it below
        # msgText = MIMEText('<b>Some <i>HTML</i> text</b> and an image.<br><img src="cid:image1"><br>Nifty!', 'html')
        # msgAlternative.attach(msgText)
        #
        # # This example assumes the image is in the current directory
        # fp = open('test.jpg', 'rb')
        # msgImage = MIMEImage(fp.read())
        # fp.close()
        #
        # # Define the image's ID as referenced above
        # msgImage.add_header('Content-ID', '<image1>')
        # msgRoot.attach(msgImage)

        return mime_msg

    def send_email(self, body, password=None):
        """
        Function to send email at the final step to the email body

        Params
        ------
            body: str
                HTML string
        """
        msg = body
        print("Sending email...")

        # password = getpass.getpass()
        if password is None:
            password = self.config['source_email_pwd']

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", port=465, context=context) as server:
            server.login(msg["From"], password)
            server.sendmail(msg["From"], msg["To"].split(','), msg.as_string())

    def run(self, get_quote=True, get_image=True):
        """
        Run function to bring it all together

        Params:
            get_quote: bool

        """
        wod_text = self.scrape_workout()

        mg_client = self.access_mongodb()

        if get_quote:
            author, quote = self.get_random_quote(mongo_client=mg_client)

        email_text = self.generate_email_text(wod_text=wod_text, get_quote=get_quote, author=author, quote=quote)
        mime_text = self.prepare_mime_format(body = email_text, embed_image=get_image)

        self.send_email(body = mime_text)
        print("Successfully sent {} daily WOD email".format(self.date))

if __name__ == "__main__":

    with open(get_config_path("mongo_private.json")) as mongo_json:
        mongo_config = json.load(mongo_json)

    with open(get_config_path("cf_email_private.json")) as cf_email_json:
        email_config = json.load(cf_email_json)

    rn = CF_email(email_config=email_config, mongo_config=mongo_config)
    rn.run(get_image=True, get_quote=True)

    # with open(get_import_path("new_quotes.json")) as quotes_js:
    #     quotes_json = json.load(quotes_js)
    #
    # update_quote_db(mongo_config=mongo_config, quotes_import_json=quotes_json)
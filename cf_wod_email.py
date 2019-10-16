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

import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
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
    config: json
        json config with email address list
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

        wod_class = "sf1KM0yG_abhAm_QFxP7E" # the tag is defined for location of WOD
        repo = soup.find(class_=wod_class)
        wod_tag =  "_6zX5t4v71r1EQ1b1O0nO2 LerElceKJ8FwMmx8LCILf" # tag for where WOD is stored
        workout_bs4_element = repo.find(class_=wod_tag) #beautiful soup element

        return workout_bs4_element

    def get_random_quote(self, user=None, password=None, mongo_url=None):
        """Function to get random quote from database for the email

        Returns
        -------
            author, quote: str, str
        """
        # initialise the database

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

        # library.list_collection_names()  # lists collection names
        # type(library['quotes'])  # type pymongo.collection.Collection

        # quote_dict = {author: quote}
        return author, quote

    def generate_email_text(self, wod_text, author, quote):
        """Function to put together message for email"""

        msg_body_html = "WOD for {date}".format(date=self.date) + "<p>"
        msg_body_html += str(wod_text) + "<p>"
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

        message = MIMEMultipart("alternative")
        message["Subject"] = " ".join([self.date, self.config['subject']])
        message["From"] = self.config['source_email']
        message["To"] = self.config['email_distrib_list']

        msg_body_html = body

        html_obj = MIMEText(msg_body_html, "html")
        message.attach(html_obj)

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", port=465, context=context) as server:
            server.login(message["From"], password)
            server.sendmail(message["From"], message["To"].split(','), message.as_string())

    def run(self):
        """Run function to bring it all together"""
        wod_text = self.scrape_workout()
        author, quote = self.get_random_quote()

        email_text = self.generate_email_text(wod_text=wod_text, author=author,
                                              quote=quote)

        self.send_email(body = email_text)
        print("Successfully sent {} daily WOD email".format(self.date))

if __name__ == "__main__":

    with open('/Users/philip_p/python/projects/config/cf_email_private.json') as json_file:
        email_config = json.load(json_file)

    with open('/Users/philip_p/python/projects/config/mongo_private.json') as json_file:
        mongo_config = json.load(json_file)

    rn = CF_email(email_config=email_config, mongo_config=mongo_config)
    rn.run()

    # with open('/Users/philip_p/python/projects/import/quotes_db.json') as json_file:
    #     quotes_json = json.load(json_file)
    #
    # update_quote_db(mongo_config=mongo_config, quotes_import_json=quotes_json)


# msg_body = "WOD for {date}: \n {workout}. \n \n {quote} - {person}".format(date=date_today_str,
#                                                                                  workout=repo_contents[0],
#                                                                                  quote = random_quote,
#                                                                                  person= random_quote_key)
# msg_body_html = "WOD for <strong>{date}</strong>:" \
#                 "<p>" \
#                 " <strong> {workout} </strong>" \
#                 "</p>" \
#                 "<br> <br> <strong> {quote} </strong> " \
#                 "<em> {person} </em>".format(date=date_today_str, workout=repo_contents[0],
#                                                quote = random_quote, person= random_quote_key)
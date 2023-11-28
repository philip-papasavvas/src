"""
Created on: 4 April 2021
Created by: Philip P
"""

import numpy as np
import pandas as pd

num_customers = 1000
num_items = 50
max_num_items = 5
selection = np.random.random_integers(
    low=0,
    high=num_items,
    size=(num_customers, num_items))

# not all customers will be buying close to 100 items, so muddy the picture
# slightly
np.random.seed(0)
random_customer_selection = np.random.choice(a=selection.shape[0], size=45,
                                             replace=False)
print(f"Customers \n {random_customer_selection} \n will have products "
      f"removed from their basket, to be in keeping with usual shopping experience")

for i in random_customer_selection:
    item_nums_to_remove = np.random.choice(
        a=selection.shape[1], size=np.random.randint(low=5, high=60))
    for item_to_remove in item_nums_to_remove:
        selection[i][item_to_remove] = 0


items_df = pd.DataFrame(
    data=selection,
    index=[f'customer_{i}' for i in range(1, num_customers + 1)])
items_df.index.name = 'customer'
items_df.reset_index(inplace=True)

items_pvt = pd.melt(items_df,
                    id_vars='customer',
                    value_vars=items_df.columns[1:],
                    value_name='item')
items_pvt.drop('variable', axis=1, inplace=True)
items_pvt['product_count'] = 1
# put in a dummy
items_pvt = items_pvt.loc[items_pvt['item'] != 0].copy(True)

product_count_df = pd.pivot_table(
    items_pvt,
    values='product_count',
    index=['customer', 'item'],
    aggfunc='count').reset_index()
product_count_df.columns = ['customer', 'item_num', 'quantity']

print(f"{product_count_df['item_num'].nunique()} unique items in the baskets"
      f" of {num_customers} customers. \n To find which customers purchased"
      f" the most similar items, we need an algorithm to do this...")


def split_customers(basket_df: pd.DataFrame,
                    num_customer_sets: int = None,
                    verbose: bool = True) -> dict:
    """
    Split the customers in the above data into sets buying similar items.
    For example if two customers buy 3 items each, try to get a set of customers
    where there is some overlap.

    Parameters
    ----------
    basket_df
       pd.DataFrame containing the columns ['customer', 'item_num', 'quantity']
    num_customer_sets
        Target number of sets to fit
    verbose
        If True, print status as it finds the set

    Returns
    -------
    dict
        keys: integers for the set number
        values: dict with
            list: customer list
            num_customers: number of customers in the set
            items_count: number of unique items in the set
    """

    # put the products into an array
    product_array = pd.pivot_table(
        basket_df,
        index='customer',
        columns='item_num',
        values='quantity',
        fill_value=0)

    max_customers_in_set = product_array.shape[0] // num_customer_sets + 1

    customers = np.array(product_array.index)

    # bool to indicate if customer has been found already
    found = np.zeros(len(customers), dtype=bool)

    # find where items have been purchased
    has_item = (product_array != 0)

    # identify as having items of not with integers. instead of using 0
    # where a customer doesn't have an item, use -1, since this
    # can discriminate against customers who have a large number of items
    # in their basket
    item_array = np.zeros(has_item.shape)
    item_array[has_item] = 1
    item_array[~has_item] = -1

    # store results in dict
    cust_sets = dict()

    # record the number of customers
    cust_count = 0
    set_count = 0

    # whilst number of customers is less than the total num of customers
    while cust_count < len(customers):
        if verbose:
            print("-" * 20)
            print(f"Customers remaining: {np.all(item_array != 0, axis=1).sum()}")
            print("-" * 20)

        # count items
        item_count = np.sum(item_array >= 0, axis=1)

        # find a basket with the least number of items
        # store the customer name in a list, get the basket items
        item_loc = np.where(item_count == item_count.min())[0][0]
        set_cust = [customers[item_loc]]
        basket_cust_reference = item_array[item_loc, :].copy()

        # go into the item array and set the basket to zero, to avoid
        # a match with itself
        item_array[item_loc, :] = 0

        # indicate customer has been 'found'
        found[item_loc] = True

        # increment customer set
        cust_count += 1

        # populate the set until the stopping conditions are met
        while (len(set_cust) < (max_customers_in_set - 1)) & (cust_count < len(customers)):
            # find nearest basket
            dist = np.dot(item_array, basket_cust_reference)

            # find location of closest basket, that hasn't already been found
            basket_loc = np.where(dist == dist[~found].max())[0][0]

            # add the customer to the set
            set_cust += [customers[basket_loc]]

            # get the reference portfolio
            basket_cust_reference = item_array[basket_loc, :].copy()
            item_array[basket_loc, :] = 0
            found[basket_loc] = True

            cust_count += 1

        # increment set number
        set_count += 1
        if verbose:
            print(f"Customer set count: {set_count}")

        # get count of number of items in a basket
        item_cnt_set = pd.pivot_table(
            basket_df.loc[basket_df['customer'].isin(set_cust)],
            index='customer',
            values='item_num',
            aggfunc='count').values.max()

        if verbose:
            print(f'Number of customers in set: {len(set_cust)}')
            print(f'Number of items {item_cnt_set}')

        # store results in dict
        cust_sets[set_count] = {'customers': set_cust,
                                'number_customers': len(set_cust),
                                'item_count': item_cnt_set}

    return cust_sets


res = split_customers(basket_df=product_count_df,
                      num_customer_sets=10)

group_1_split_df = product_count_df.loc[
    product_count_df['customer'].isin(res[1]['customers'])]

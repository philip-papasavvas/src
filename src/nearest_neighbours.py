"""
Created on: 4 April 2021
Created by: Philip P

This script simulates a shopping scenario by generating a relationship between customers
and items in a store. It creates a DataFrame that represents the quantity of each item
selected by each customer, applying a realistic randomness to the selection and quantity.
"""
import numpy as np
import pandas as pd


class CustomerItemsGenerator:
    """
    A class used to generate a DataFrame representing customer-item relationships.

    Attributes
    ----------
    num_customers : int
        The number of customers.
    num_items : int
        The number of items.
    max_num_items : int
        The maximum number of items a customer can have.
    selection : np.ndarray
        The 2D array representing the selection of items by customers.
    product_count_df : pd.DataFrame
        The DataFrame representing the count of each product for each customer.

    Methods
    -------
    generate()
        Generates the customer-item selection and the product count DataFrame.
    """
    def __init__(self, num_customers: int, num_items: int, max_num_items: int):
        self.num_customers = num_customers
        self.num_items = num_items
        self.max_num_items = max_num_items
        self.selection = None
        self.product_count_df = None

    def generate(self):
        """Generates the customer-item selection and the product count DataFrame."""
        self._generate_customer_item_selection()
        self._apply_realistic_shopping_experience()
        self._generate_product_count_df()

    def _generate_customer_item_selection(self):
        """Generates a 2D array representing the selection of items by customers."""
        self.selection = np.random.randint(low=0,
                                           high=self.num_items,
                                           size=(self.num_customers, self.num_items))

    def _apply_realistic_shopping_experience(self):
        """Applies a realistic shopping experience to the selection."""
        np.random.seed(0)
        random_customer_selection = np.random.choice(a=self.selection.shape[0],
                                                     size=min(45, self.selection.shape[0]),
                                                     replace=False)
        for i in random_customer_selection:
            item_nums_to_remove = np.random.choice(a=self.selection.shape[1],
                                                   size=np.random.randint(low=5, high=60)
                                                   )
            self.selection[i][item_nums_to_remove] = 0

    def _generate_product_count_df(self):
        """Generates a DataFrame representing the count of each product for each customer."""
        items_df = pd.DataFrame(data=self.selection,
                                index=[f'customer_{i}' for i in range(1, self.num_customers + 1)])
        items_df.index.name = 'customer'
        items_df.reset_index(inplace=True)

        items_pvt = pd.melt(items_df, id_vars='customer', value_vars=items_df.columns[1:], value_name='item')
        items_pvt.drop('variable', axis=1, inplace=True)
        items_pvt['product_count'] = 1
        items_pvt = items_pvt.loc[items_pvt['item'] != 0].copy(True)

        self.product_count_df = pd.pivot_table(items_pvt, values='product_count', index=['customer', 'item'], aggfunc='count').reset_index()
        self.product_count_df.columns = ['customer', 'item_num', 'quantity']


if __name__ == '__main__':
    # example of using the class
    generator = CustomerItemsGenerator(num_customers=10, num_items=10, max_num_items=5)
    generator.generate()
    print(generator.product_count_df)

"""Generate a synthetic retail transaction dataset for market basket analysis.

Context: a Kenyan FMCG retail network (supermarkets and neighbourhood dukas)
selling dairy, bakery, breakfast, staples, meats, snacks, beverages and
household goods.

How baskets are built (so the analysis has real structure to discover):
  1. Each basket gets 1-2 shopping "missions" (breakfast restock, dinner cook,
     party/snack run, top-up, cleaning, deli/BBQ ...). Each mission adds its
     products with mission-specific probabilities - this is what creates the
     associations (e.g. tea -> sugar -> fresh milk).
  2. A few "impulse" items are added at random, weighted by general popularity,
     which adds realistic noise.
  3. Supermarket baskets are bigger and include more premium items; duka
     baskets are small top-ups.

All data is randomly generated. No real company or customer data is included.
Usage: python data/generate_transactions.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
OUT = Path(__file__).parent
N_BASKETS = 15_000

# product -> (category, unit price KES, base popularity for impulse picks)
PRODUCTS = {
    "Fresh Milk 500ml": ("Dairy", 65, 10), "Long-life Milk 1L": ("Dairy", 140, 4),
    "Natural Yoghurt 500ml": ("Dairy", 180, 4), "Flavoured Yoghurt 250ml": ("Dairy", 90, 6),
    "Cheddar Cheese 200g": ("Dairy", 520, 2), "Mozzarella 200g": ("Dairy", 560, 1.5),
    "Butter 250g": ("Dairy", 390, 3), "Fresh Cream 250ml": ("Dairy", 260, 1),
    "Ice Cream 1L": ("Frozen", 650, 2), "Ice Cream Cones": ("Frozen", 220, 0.8),
    "White Bread 400g": ("Bakery", 65, 9), "Brown Bread 400g": ("Bakery", 70, 4),
    "Burger Buns 6pk": ("Bakery", 180, 1.5), "Queen Cakes 6pk": ("Bakery", 150, 2),
    "Tea Leaves 250g": ("Breakfast", 160, 4), "Instant Coffee 100g": ("Breakfast", 450, 1.5),
    "Sugar 1kg": ("Breakfast", 190, 5), "Breakfast Cereal 500g": ("Breakfast", 520, 2),
    "Granola 400g": ("Breakfast", 610, 1), "Strawberry Jam 450g": ("Breakfast", 350, 1.5),
    "Peanut Butter 400g": ("Breakfast", 420, 1.5), "Margarine 500g": ("Breakfast", 230, 3),
    "Eggs Tray (30)": ("Staples", 480, 3), "Maize Flour 2kg": ("Staples", 210, 6),
    "Rice 2kg": ("Staples", 380, 3), "Cooking Oil 1L": ("Staples", 340, 4),
    "Spaghetti 500g": ("Staples", 160, 2), "Pasta Sauce 400g": ("Staples", 280, 1),
    "Tomatoes 1kg": ("Fresh Produce", 120, 5), "Onions 1kg": ("Fresh Produce", 110, 5),
    "Beef Sausages 500g": ("Deli & Meat", 420, 2.5), "Smoked Bacon 250g": ("Deli & Meat", 480, 1),
    "Chicken Wings 1kg": ("Deli & Meat", 690, 1.5), "BBQ Sauce 375ml": ("Deli & Meat", 290, 0.6),
    "Potato Crisps 150g": ("Snacks", 150, 4), "Salted Peanuts 200g": ("Snacks", 130, 2.5),
    "Soda 2L": ("Beverages", 210, 5), "Fruit Juice 1L": ("Beverages", 260, 3),
    "Bottled Water 1L": ("Beverages", 80, 4),
    "Washing Powder 1kg": ("Household", 330, 2.5), "Bar Soap 800g": ("Household", 180, 3),
    "Toilet Paper 10pk": ("Household", 450, 2.5), "Diapers 40pk": ("Baby", 1450, 1),
    "Baby Wipes": ("Baby", 280, 0.8),
}

# mission -> (weight in supermarket, weight in duka, {product: probability})
MISSIONS = {
    "tea_time":   (12, 30, {"Tea Leaves 250g": .75, "Sugar 1kg": .55, "Fresh Milk 500ml": .85,
                            "White Bread 400g": .45, "Queen Cakes 6pk": .15, "Margarine 500g": .2}),
    "breakfast":  (14, 12, {"White Bread 400g": .55, "Brown Bread 400g": .3, "Butter 250g": .35,
                            "Margarine 500g": .3, "Strawberry Jam 450g": .35, "Peanut Butter 400g": .25,
                            "Eggs Tray (30)": .4, "Fresh Milk 500ml": .5}),
    "cereal":     (7, 2, {"Breakfast Cereal 500g": .85, "Long-life Milk 1L": .55, "Fresh Milk 500ml": .3,
                          "Granola 400g": .2, "Natural Yoghurt 500ml": .3}),
    "yoghurt_bowl": (5, 1, {"Natural Yoghurt 500ml": .8, "Granola 400g": .6, "Fruit Juice 1L": .2}),
    "dinner_cook": (16, 20, {"Maize Flour 2kg": .5, "Rice 2kg": .35, "Cooking Oil 1L": .5,
                             "Tomatoes 1kg": .7, "Onions 1kg": .7}),
    "pasta_night": (6, 1, {"Spaghetti 500g": .9, "Pasta Sauce 400g": .65, "Cheddar Cheese 200g": .3,
                           "Mozzarella 200g": .35, "Onions 1kg": .2}),
    "bbq_party":  (6, 2, {"Beef Sausages 500g": .6, "Chicken Wings 1kg": .5, "Burger Buns 6pk": .45,
                          "BBQ Sauce 375ml": .35, "Soda 2L": .6, "Potato Crisps 150g": .4,
                          "Cheddar Cheese 200g": .2}),
    "snack_run":  (8, 12, {"Potato Crisps 150g": .55, "Salted Peanuts 200g": .4, "Soda 2L": .55,
                           "Bottled Water 1L": .3, "Flavoured Yoghurt 250ml": .25}),
    "dessert":    (4, 1, {"Ice Cream 1L": .9, "Ice Cream Cones": .45, "Fresh Cream 250ml": .2,
                          "Queen Cakes 6pk": .2}),
    "fry_up":     (5, 2, {"Smoked Bacon 250g": .7, "Beef Sausages 500g": .5, "Eggs Tray (30)": .65,
                          "Brown Bread 400g": .35, "Butter 250g": .25}),
    "cleaning":   (8, 8, {"Washing Powder 1kg": .7, "Bar Soap 800g": .6, "Toilet Paper 10pk": .5}),
    "baby":       (4, 2, {"Diapers 40pk": .9, "Baby Wipes": .6, "Long-life Milk 1L": .25}),
    "kids_lunch": (5, 3, {"Flavoured Yoghurt 250ml": .7, "Fruit Juice 1L": .5, "White Bread 400g": .3,
                          "Peanut Butter 400g": .3}),
}

names = list(PRODUCTS)
pop = np.array([PRODUCTS[p][2] for p in names], dtype=float)
pop /= pop.sum()
mission_names = list(MISSIONS)

rows = []
dates = pd.date_range("2026-01-01", "2026-06-30", freq="D")
for b in range(N_BASKETS):
    store = RNG.choice(["Supermarket", "Duka"], p=[.6, .4])
    weights = np.array([MISSIONS[m][0 if store == "Supermarket" else 1] for m in mission_names], float)
    weights /= weights.sum()
    n_missions = RNG.choice([1, 2, 3], p=[.55, .35, .10] if store == "Supermarket" else [.85, .14, .01])
    basket = set()
    for m in RNG.choice(mission_names, size=n_missions, replace=False, p=weights):
        for prod, prob in MISSIONS[m][2].items():
            if RNG.random() < prob:
                basket.add(prod)
    n_impulse = RNG.poisson(1.6 if store == "Supermarket" else 0.5)
    basket.update(RNG.choice(names, size=n_impulse, p=pop))
    if not basket:                                   # every basket has at least one item
        basket.add(RNG.choice(names, p=pop))
    date = RNG.choice(dates)
    for prod in basket:
        cat, price, _ = PRODUCTS[prod]
        rows.append(dict(invoice_id=f"INV{b:06d}", date=pd.Timestamp(date).date(),
                         store_type=store, product=prod, category=cat,
                         quantity=int(RNG.choice([1, 1, 1, 2, 3])), unit_price_kes=price))

df = pd.DataFrame(rows).sort_values(["invoice_id", "category", "product"])
df.to_csv(OUT / "transactions.csv", index=False)
print(f"{df.invoice_id.nunique():,} baskets | {len(df):,} line items | {df['product'].nunique()} products")
